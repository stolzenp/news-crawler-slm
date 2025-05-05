from datasets import load_from_disk, DatasetDict

# load dataset from a local directory
dataset = load_from_disk("/vol/tmp/stolzenp/training/shrinked_split_dataset_cleaned_filtered_24k")

# set Hugging Face repo ID
repo_id = "stolzenp/fundus-cleaned-filtered-62K"

# extract reference features from train split in case not all splits have the same features due to missing values
reference_features = dataset["train"].features

# align features for all splits
aligned_dataset = DatasetDict({
    split: ds.cast(reference_features)
    for split, ds in dataset.items()
})

# push to Hugging Face Hub
aligned_dataset.push_to_hub(repo_id)

print(f"Dataset uploaded to https://huggingface.co/datasets/{repo_id}")
