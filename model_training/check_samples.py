from datasets import load_from_disk
from utils import get_args_from_config

model_args = get_args_from_config("model_training_settings")
dataset_dir = model_args["split_dataset_directory"]
dataset = load_from_disk(dataset_dir)

sample = dataset["train"][18051]

print(sample["json"])