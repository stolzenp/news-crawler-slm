import torch, math
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from datasets import load_from_disk
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from utils import get_args_from_config

model_args = get_args_from_config("model_training_settings")
model_name = model_args["model_name_or_path"]

dataset_dir = model_args["tokenized_dataset_directory"]

dataset = load_from_disk(dataset_dir)

# Initialize DDP
dist.init_process_group(backend="nccl")

# Get local rank (which GPU this process should use)
local_rank = dist.get_rank()
torch.cuda.set_device(local_rank)

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Move model to the correct GPU
model = model.to(local_rank)

# Wrap with DistributedDataParallel (DDP)
model = DDP(model, device_ids=[local_rank])

training_args = SFTConfig(
    output_dir="/vol/tmp/stolzenp/results",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    logging_dir = "/vol/tmp/stolzenp/logs",
    logging_steps=10,
    learning_rate=2e-3,
    dataset_kwargs = {"skip_prepare_dataset": True}
)

# Compute Perplexity (PPL)
# maybe BLEU or ROUGE instead?
def compute_metrics(eval_pred):
    # get logits
    logits, labels = eval_pred

    # shift the logits and labels for padding token handling
    logits = torch.tensor(logits)
    labels = torch.tensor(labels)

    # shift the logits to ignore padding tokens (flatten for loss computation)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=1)

    # get mean NLL with flatten logits and labels
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    # Perplexity is the exponential of the mean NLL
    perplexity = math.exp(loss.item())
    return {"perplexity": perplexity}

lora_config =  LoraConfig(
        r=8,
        target_modules=["x_proj", "embeddings", "in_proj", "out_proj"],
        task_type="CAUSAL_LM",
        bias="none"
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    args=training_args,
    peft_config=lora_config,
    train_dataset=dataset["train"],
    eval_dataset=dataset["val"],
    compute_metrics=compute_metrics,
)

trainer.train()

results = trainer.evaluate(dataset["test"])
print("Final Evaluation:", results)