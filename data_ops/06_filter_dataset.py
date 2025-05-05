import re
from datasets import load_from_disk
from datasets import DatasetDict
import os

# Define thresholds
first_list_threshold = 16000
other_lists_threshold = 4000

# Storage for positions to remove, grouped by split
positions_to_remove = {"train": set(), "val": set(), "test": set()}  # Adjust splits if necessary

# Read the file
with open("/vol/fob-vol4/mi17/stolzenp/news-crawler-slm/token_statistics_cleaned_html.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

# Track list index
list_index = 0

for line in lines:
    match = re.search(r"Token Count:\s*(\d+),\s*Split:\s*(\w+),\s*Position:\s*(\d+)", line)
    if match:
        token_count = int(match.group(1))
        split = match.group(2)
        position = int(match.group(3))

        # Determine threshold based on list index
        threshold = first_list_threshold if list_index == 2 else other_lists_threshold

        if token_count > threshold:
            positions_to_remove[split].add(position)

    # Detect list boundaries (assuming lists are separated by empty lines)
    elif line.strip() == "":
        list_index += 1

# Load Hugging Face dataset
dataset_path = "/vol/tmp/stolzenp/training/split_dataset_cleaned"  # Change to actual dataset path
dataset = load_from_disk(dataset_path)

# Filter dataset by keeping only entries NOT in positions_to_remove
filtered_dataset = {}
for split in dataset.keys():
    if split in positions_to_remove:
        filtered_dataset[split] = dataset[split].filter(lambda _, idx: idx not in positions_to_remove[split],
                                                         with_indices=True)
    else:
        filtered_dataset[split] = dataset[split]

split_dataset_cleaned_filtered_dir = "/vol/tmp/stolzenp/training/split_dataset_cleaned_filtered_16k_4k"

filtered_dataset = DatasetDict(filtered_dataset)

# Save filtered dataset
os.makedirs(split_dataset_cleaned_filtered_dir, exist_ok=True)
filtered_dataset.save_to_disk(split_dataset_cleaned_filtered_dir)
