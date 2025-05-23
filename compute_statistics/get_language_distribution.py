import os
from collections import Counter

from datasets import load_from_disk

from common.utils import get_args_from_config

# assign relevant arguments
stats_args = get_args_from_config("statistics_settings")
output_dir = stats_args["output_dir"]
dataset_dir = stats_args["dataset_directory"]
language_stats_file = stats_args["language_stats_file"]

# set path
os.makedirs(output_dir, exist_ok=True)
language_stats_file = f"{output_dir}/{language_stats_file}"

# load dataset
dataset = load_from_disk(dataset_dir)

# compute and save language statistics
with open(language_stats_file, "w", encoding="utf-8") as f:
    for split in dataset.keys():
        lang_counts = Counter(dataset[split]["language"])
        f.write(f"Language distribution in {split}:\n")
        for lang, count in lang_counts.most_common():
            f.write(f"  {lang}: {count}\n")
        f.write("\n")

print(f"Results saved to {language_stats_file}")
