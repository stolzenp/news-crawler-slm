from unsloth import FastLanguageModel
from transformers import Qwen2ForCausalLM, AutoModelForCausalLM
import torch
import math
from datasets import load_from_disk, concatenate_datasets
from trl import SFTTrainer
from transformers import TrainingArguments
from transformers import modeling_outputs
from unsloth import is_bfloat16_supported
from typing import Optional, Tuple, List, Union
import argparse

from utils import get_args_from_config

# subclass for applying contrastive loss approach in model forward
class Qwen2ForCausalLMWithCL(Qwen2ForCausalLM):

    def __init__(self, config):
        super().__init__(config)
        self.embed_dim = config.hidden_size

        # Initialize weights and apply final processing
        self.post_init()

    def compute_contrastive_loss(self, score_matrix, margin):
        '''
           margin: predefined margin to push similarity score away
           score_matrix: bsz x seqlen x seqlen; cosine similarity matrix
           input_ids: bsz x seqlen
        '''
        bsz, seqlen, _ = score_matrix.size()
        gold_score = torch.diagonal(score_matrix, offset=0, dim1=1, dim2=2)  # bsz x seqlen
        gold_score = torch.unsqueeze(gold_score, -1)
        assert gold_score.size() == torch.Size([bsz, seqlen, 1])
        difference_matrix = gold_score - score_matrix
        assert difference_matrix.size() == torch.Size([bsz, seqlen, seqlen])
        loss_matrix = margin - difference_matrix  # bsz x seqlen x seqlen
        loss_matrix = torch.nn.functional.relu(loss_matrix)
        cl_loss = torch.mean(loss_matrix)
        return cl_loss

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            num_logits_to_keep: int = 0,
            # we set 0.5 as default to avoid also subclassing trainer
            margin: float = 0.5, # Su et al. achieved best results with this value in their paper
            **loss_kwargs,
    ) -> Union[Tuple, modeling_outputs.CausalLMOutputWithPast]:

        bsz, seqlen = input_ids.size()
        outputs = super().forward(input_ids, attention_mask=attention_mask, labels=labels, **loss_kwargs, output_hidden_states=True)
        logits = outputs.logits

        assert logits.size() == torch.Size([bsz, seqlen, self.vocab_size])
        last_hidden_states = outputs.hidden_states[-1]
        assert last_hidden_states.size() == torch.Size([bsz, seqlen, self.embed_dim])
     #   mle_loss = CrossEntropyLoss(logits.view(-1, self.vocab_size), labels.view(-1))

        norm_rep = last_hidden_states / last_hidden_states.norm(dim=2, keepdim=True)
        cosine_scores = torch.matmul(norm_rep, norm_rep.transpose(1, 2))
        assert cosine_scores.size() == torch.Size([bsz, seqlen, seqlen])
        cl_loss = self.compute_contrastive_loss(cosine_scores, margin)

        new_loss = outputs.loss + cl_loss

        print(f"old_loss: {outputs.loss}")
        print(f"cl_loss: {cl_loss}")
        print(f"new_loss: {new_loss}")

        return modeling_outputs.CausalLMOutputWithPast(
            loss=new_loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

parser = argparse.ArgumentParser(description="Finetune model")
parser.add_argument("type", help="Input type")

args = parser.parse_args()
given_type = args.type
altered_type = args.type.replace("_", "")

model_args = get_args_from_config("model_training_settings")

dataset_dir = model_args["split_dataset_directory"]
model_name = model_args["model_name_or_path"]

max_seq_length = 32768 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

original_from_pretrained = AutoModelForCausalLM.from_pretrained

def custom_from_pretrained(model_name, *args, **kwargs):
    if "qwen2" in model_name or "ReaderLM-v2" in model_name:
        print(f"⚡ Loading CustomQwen2Model instead of Qwen2ForCausalLM for {model_name}")
        return Qwen2ForCausalLMWithCL.from_pretrained(model_name, *args, **kwargs)
    return original_from_pretrained(model_name, *args, **kwargs)

# Apply the patch
AutoModelForCausalLM.from_pretrained = custom_from_pretrained

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

print(model)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

print(model)

prompt = """
Input:
{}

Output:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    inputs       = examples["html"]
    outputs      = examples[given_type]
    texts = []
    for input, output in zip(inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = prompt.format(input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass

dataset = load_from_disk(dataset_dir)
dataset = dataset.map(formatting_prompts_func, batched = True,)

PAD_TOKEN = tokenizer.pad_token # Get PAD_TOKEN

# Compute Perplexity (PPL)
# maybe BLEU or ROUGE instead?
def compute_metrics(eval_pred):
    # Get logits and labels
    logits, labels = eval_pred

    # Convert to tensors
    logits = torch.as_tensor(logits)
    labels = torch.as_tensor(labels)

    # Shift logits and labels for causal LM
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    # Flatten for loss computation
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1)

    # Define loss function (ignore padding tokens)
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    loss = loss_fct(shift_logits, shift_labels)

    # Compute perplexity
    perplexity = math.exp(loss.item())

    return {"perplexity": perplexity}

train_dataset = concatenate_datasets([dataset["train"], dataset["val"]])

trainer = SFTTrainer(
    model = model,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
    train_dataset = train_dataset,
    eval_dataset=dataset["test"],
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 1,
        per_device_eval_batch_size = 1,
        gradient_accumulation_steps = 1,
      #  warmup_steps = 105,
        num_train_epochs = 1, # Set this for 1 full training run.
        learning_rate = 4e-5,
        eval_strategy = "epoch",
     #   eval_steps = 10,
    #    eval_accumulation_steps = 10,
        logging_dir = f"/vol/tmp/stolzenp/training/ReaderLM-v2_24k+8k_cl_loss/logs/{altered_type}",
        save_strategy = "epoch",
    #    save_steps=600,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 100,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = f"/vol/tmp/stolzenp/training/ReaderLM-v2_24k+8k_cl_loss/results/{altered_type}",
        report_to = ["tensorboard"], # Use this for WandB etc
        load_best_model_at_end = True,
        metric_for_best_model = "perplexity",
    ),
)


trainer_stats = trainer.train()

#results = trainer.evaluate(dataset["test"])
#print("Final Evaluation:", results)