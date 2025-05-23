from datasets import DatasetDict, load_from_disk

from common.utils import get_args_from_config

# get dataset arguments from the config file
dataset_args = get_args_from_config("data_extraction_settings")

# set dataset variables
dataset_path = dataset_args["dataset_directory"]
hf_dataset_path = dataset_args["hf_dataset_path"]

# load dataset from a local directory
dataset = load_from_disk(dataset_path)

# extract reference features from train split in case not all splits have the same features due to missing values
reference_features = dataset["train"].features

# align features for all splits
aligned_dataset = DatasetDict({split: ds.cast(reference_features) for split, ds in dataset.items()})

# push dataset to Hugging Face Hub
aligned_dataset.push_to_hub(hf_dataset_path)

print(f"Dataset uploaded to https://huggingface.co/datasets/{hf_dataset_path}")
