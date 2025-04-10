from datasets import load_from_disk
from datasets import DatasetDict, Dataset
import os

# Function to limit each publisher to at most 50 samples
def limit_samples_per_publisher(dataset, max_samples=50):
    grouped = {}

    for sample in dataset:
        publisher = sample["publisher"]
        if publisher not in grouped:
            grouped[publisher] = []
        if len(grouped[publisher]) < max_samples:
            grouped[publisher].append(sample)

    # Flatten the grouped dictionary back into a list
    new_data = [sample for samples in grouped.values() for sample in samples]

    return Dataset.from_list(new_data)

# Define thresholds
samples_per_publisher = 10

dataset_path = "/vol/tmp/stolzenp/training/split_dataset_cleaned_filtered_24k"
dataset = load_from_disk(dataset_path)

# Print splits and sizes
for split, data in dataset.items():
    print(f"Split: {split}, Size: {len(data)}")

shrinked_split_dataset_cleaned_filtered_dir = "/vol/tmp/stolzenp/training/shrinked_split_dataset_cleaned_filtered_24k"
shrinked_dataset = {}

for split in dataset.keys():
    if split == "train":
        shrinked_dataset[split] = dataset[split]
        continue
    shrinked_dataset[split] = limit_samples_per_publisher(dataset[split], samples_per_publisher)

shrinked_dataset = DatasetDict(shrinked_dataset)

# Print splits and sizes
for split, data in shrinked_dataset.items():
    print(f"Split: {split}, Size: {len(data)}")

# Save filtered dataset
os.makedirs(shrinked_split_dataset_cleaned_filtered_dir, exist_ok=True)
shrinked_dataset.save_to_disk(shrinked_split_dataset_cleaned_filtered_dir)
