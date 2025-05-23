import os

from datasets import load_from_disk

from common.utils import get_args_from_config

# assign relevant arguments
stats_args = get_args_from_config("statistics_settings")
output_dir = stats_args["output_dir"]
dataset_dir = stats_args["dataset_directory"]
dataset_stats_file = stats_args["dataset_stats_file"]

# set path
dataset_stats_file = f"{output_dir}/{dataset_stats_file}"
os.makedirs(os.path.dirname(dataset_stats_file), exist_ok=True)

# load dataset
dataset = load_from_disk(dataset_dir)

# save and print dataset stats
with open(dataset_stats_file, "w") as f:
    for split in dataset.keys():
        line = f"{split}: {dataset[split].num_rows} samples\n"
        print(line, end="")
        f.write(line)

print(f"\nSaved dataset sample counts to {dataset_stats_file}")
