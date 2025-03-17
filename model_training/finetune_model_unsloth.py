from unsloth import FastLanguageModel
import torch
import math
from datasets import load_from_disk, concatenate_datasets
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

from utils import get_args_from_config

model_args = get_args_from_config("model_training_settings")

dataset_dir = model_args["split_dataset_directory"]
model_name = model_args["model_name_or_path"]

max_seq_length = 32768 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

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

prompt = """
Input:
{}

Output:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    inputs       = examples["html"]
    outputs      = examples["plain_text"]
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
        logging_dir = "/vol/tmp/stolzenp/training/qwen2.5-1.5b_24k+8k/logs",
        save_strategy = "epoch",
    #    save_steps=600,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 100,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "/vol/tmp/stolzenp/training/qwen2.5-1.5b_24k+8k/results/json",
        report_to = ["tensorboard"], # Use this for WandB etc
        load_best_model_at_end = True,
        metric_for_best_model = "perplexity",
    ),
)

#trainer_stats = trainer.train()

results = trainer.evaluate(dataset["test"])
print("Final Evaluation:", results)