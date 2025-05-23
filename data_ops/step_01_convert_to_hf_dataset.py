import json
import os

from datasets import Dataset

from common.utils import get_args_from_config

# assign arguments to variables
data_args = get_args_from_config("data_ops_settings")
dataset_path = data_args["source_dataset_path"]
output_dir = data_args["target_dataset_directory"]

# reading the .txt dataset file and parse the JSON entries
data = []
with open(dataset_path, 'r',  encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))

# first converting the list of dictionaries to a dictionary of lists
columns = {}
for entry in data:
    for key, value in entry.items():
        if key not in columns:
            columns[key] = []
        columns[key].append(value)

# converting the dictionary of lists into a Hugging Face Dataset
dataset = Dataset.from_dict(columns)

# save Hugging Face Dataset and create directories if needed
os.makedirs(output_dir, exist_ok=True)
dataset.save_to_disk(output_dir)