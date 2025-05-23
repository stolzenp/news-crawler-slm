import os
import re

from datasets import DatasetDict, load_from_disk

from common.utils import get_args_from_config

data_ops = get_args_from_config("data_ops_settings")
html_token_threshold = data_ops["html_token_threshold"]
target_token_threshold = data_ops["target_token_threshold"]
token_stats_file = data_ops["token_stats_file"]
cleaned_dataset_dir = data_ops["clean_dataset_directory"]
filtered_dataset_dir = data_ops["filtered_dataset_directory"]

# load dataset
dataset = load_from_disk(cleaned_dataset_dir)

# storage for positions to remove, grouped by split
positions_to_remove = {split: set() for split in dataset.keys()}

# read the file
with open(token_stats_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

# track list index
list_index = 0

# go through token statistics for dataset
for line in lines:
    match = re.search(r"Token Count:\s*(\d+),\s*Split:\s*(\w+),\s*Position:\s*(\d+)", line)
    if match:
        token_count = int(match.group(1))
        split = match.group(2)
        position = int(match.group(3))

        # determine the threshold to apply based on the list index
        threshold = html_token_threshold if list_index == 2 else target_token_threshold

        if token_count > threshold:
            positions_to_remove[split].add(position)

    # detect list boundaries (assuming lists are separated by empty lines)
    elif line.strip() == "":
        list_index += 1

# filter dataset by keeping only entries not in positions_to_remove
filtered_dataset = {}
for split in dataset.keys():
    if split in positions_to_remove:
        filtered_dataset[split] = dataset[split].filter(
            lambda _, idx, s=split: idx not in positions_to_remove[s], with_indices=True
        )
    else:
        filtered_dataset[split] = dataset[split]

filtered_dataset = DatasetDict(filtered_dataset)

# Save filtered dataset
os.makedirs(filtered_dataset_dir, exist_ok=True)
filtered_dataset.save_to_disk(filtered_dataset_dir)
