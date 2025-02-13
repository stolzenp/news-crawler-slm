import os
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer
from utils import get_args_from_config

# get config arguments
model_args = get_args_from_config("model_training_settings")

# assign relevant arguments
dataset_dir = model_args["split_dataset_directory"]
tokenized_dataset_dir =  model_args["tokenized_dataset_directory"]
model_name = model_args["model_name_or_path"]
target_column = model_args["target_column"]

dataset = load_from_disk(dataset_dir)

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

# define mapping function
def preprocess_function(examples, max_input_length=16384, max_target_length=2048):
    inputs = tokenizer(
        examples["html"],
        max_length=max_input_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    # set global attention on first token (important for LED)
    inputs["global_attention_mask"] = torch.zeros_like(inputs["input_ids"])
    inputs["global_attention_mask"][:, 0] = 1  # global attention on first token

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples[target_column],
            max_length=max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

    inputs["labels"] = labels["input_ids"]
    return inputs

# apply tokenization to all splits
tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)

# create target directory if not existent
os.makedirs(tokenized_dataset_dir, exist_ok=True)

# save tokenized dataset for later usage
tokenized_dataset.save_to_disk(tokenized_dataset_dir)