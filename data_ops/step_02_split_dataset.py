import random
import os
from datasets import DatasetDict, load_from_disk

from common.utils import get_args_from_config

# get config arguments
data_args = get_args_from_config("data_ops_settings")

# assign relevant arguments
dataset_dir = data_args["target_dataset_directory"]
split_dataset_dir =  data_args["split_dataset_directory"]
training_split_size = data_args["training_split_size"]
seed = data_args["seed"]

# load dataset
dataset = load_from_disk(dataset_dir)

# get unique publishers
publishers = list(set(dataset['publisher']))

# shuffle with seed
random.seed(seed)
random.shuffle(publishers)

# calculate split sizes
num_publishers = len(publishers)
train_size = round(training_split_size * num_publishers)
remaining = num_publishers - train_size
val_size = test_size = remaining // 2
if remaining % 2 == 1:
    train_size += 1  # adjust train size if needed

# assign publishers to splits
train_publishers = set(publishers[:train_size])
val_publishers = set(publishers[train_size:train_size + val_size])
test_publishers = set(publishers[train_size + val_size:])

# filter dataset to get splits
train_set = dataset.filter(lambda x: x['publisher'] in train_publishers)
val_set = dataset.filter(lambda x: x['publisher'] in val_publishers)
test_set = dataset.filter(lambda x: x['publisher'] in test_publishers)

# create the split dataset
split_dataset = DatasetDict({
    "train": train_set,
    "val": val_set,
    "test": test_set,
})

# save split dataset and create directories if needed
os.makedirs(split_dataset_dir, exist_ok=True)
split_dataset.save_to_disk(split_dataset_dir)