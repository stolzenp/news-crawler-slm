import torch
import math
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

from utils import get_args_from_config

torch.cuda.empty_cache()

model_args = get_args_from_config("model_training_settings")

dataset_dir = model_args["tokenized_dataset_directory"]
model_name = model_args["model_name_or_path"]

dataset = load_from_disk(dataset_dir)

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_name)

training_args = TrainingArguments(
    output_dir="/vol/tmp/stolzenp/training/results",
    num_train_epochs=5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=1,  # simulate larger batch sizes
    learning_rate=5e-5,
    warmup_steps=105,
    max_grad_norm=1.0,
    eval_strategy = "epoch",
  #  eval_steps = 200,
    logging_dir = "/vol/tmp/stolzenp/training/logs",
    save_strategy = "epoch",
   # save_steps=600,
    logging_steps = 100,
    fp16=True,
    load_best_model_at_end = True,
    metric_for_best_model = "perplexity",
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

    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=151643)

    # get mean NLL with flatten logits and labels
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    # Perplexity is the exponential of the mean NLL
    perplexity = math.exp(loss.item())
    return {"perplexity": perplexity}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["val"],
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

model.config.use_cache = False
model.gradient_checkpointing_enable()
trainer.train()

results = trainer.evaluate(dataset["test"])
print("Final Evaluation:", results)