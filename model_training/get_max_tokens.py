import json
import os
import torch
from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoTokenizer
from utils import get_args_from_config

# get config arguments
model_args = get_args_from_config("model_training_settings")

# assign relevant arguments
dataset_dir = model_args["target_dataset_directory"]
model_name = model_args["model_name_or_path"]

dataset = load_from_disk(dataset_dir)

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

test_sample = dataset["json"][3]

#max_length_html = max(len(tokenizer.tokenize(text)) for text in tqdm(dataset["html"], desc="Processing 'html' Tokens"))

#with open("max_length.txt", "a") as f:
 #   f.write(f"Maximum token length for html column: {max_length_html}\n")

#max_length_plain = max(len(tokenizer.tokenize(text)) for text in tqdm(dataset["plain_text"], desc="Processing 'plain_text' Tokens"))

#with open("max_length.txt", "a") as f:
 #   f.write(f"Maximum token length for plain_text column: {max_length_plain}\n")

max_length_json = max(len(tokenizer.tokenize(json.dumps(text))) for text in tqdm(dataset["json"], desc="Processing 'json' Tokens"))

with open("max_length.txt", "a") as f:
    f.write(f"Maximum token length for json column: {max_length_json}\n")

print("Max token lengths saved to max_length.txt")