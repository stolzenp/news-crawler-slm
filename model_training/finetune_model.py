import torch
import numpy as np
import math
from datasets import load_from_disk
from transformers import AutoTokenizer, LEDForConditionalGeneration, Trainer, TrainingArguments

from utils import get_args_from_config

model_args = get_args_from_config("model_training_settings")

dataset_dir = model_args["tokenized_dataset_directory"]
model_name = model_args["model_name_or_path"]

dataset = load_from_disk(dataset_dir)

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = LEDForConditionalGeneration.from_pretrained(model_name)

training_args = TrainingArguments(
    output_dir="/vol/tmp/stolzenp/results",
    max_steps=1000,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,  # simulate larger batch sizes
    learning_rate=1e-4,
    warmup_steps=100,
    max_grad_norm=1.0,
    eval_strategy = "steps",
    eval_steps = 100,
    logging_dir = "/vol/tmp/stolzenp/logs",
    save_steps = 500,
    logging_steps = 100,
    save_total_limit = 3,
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

    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=1)

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
trainer.train()

results = trainer.evaluate(dataset["test"])
print("Final Evaluation:", results)