import json
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from collections import Counter
from tqdm import tqdm
import numpy as np
from datasets import load_from_disk
from transformers import AutoTokenizer
from utils import get_args_from_config

# get config arguments
model_args = get_args_from_config("model_training_settings")

# assign relevant arguments
dataset_dir = model_args["split_dataset_directory"]
model_name = model_args["model_name_or_path"]

dataset = load_from_disk(dataset_dir)

# Load tokenizer (replace with your model)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# File to save statistics
stats_file = "token_statistics_cleaned_html_24k.txt"

# Columns to analyze
columns_to_process = ["html", "plain_text", "json"]


# Function to tokenize and count tokens
def get_token_counts(samples):
    return [len(tokenizer.tokenize(entry)) for entry in tqdm(samples, desc="Tokenizing", ncols=80)]


# Aggregate statistics per column across all splits
column_token_counts = {column: [] for column in columns_to_process}

with open(stats_file, "w") as f:
    for split in ["train", "val", "test"]:
        print(f"Processing {split} split...")

        for column in columns_to_process:
            print(f"Processing column: {column}")

            # Extract texts (handling JSON conversion if needed)
            if column == "json":
                texts = [json.dumps(entry) for entry in dataset[split][column]]
            else:
                texts = dataset[split][column]

            # Tokenize and store token counts with metadata (split and position)
            token_counts = get_token_counts(texts)
            column_token_counts[column].extend(
                (token_count, split, idx) for idx, token_count in enumerate(token_counts)
            )

    # Compute statistics per column across all splits
    for column, token_data in column_token_counts.items():
        token_counts_only = [t[0] for t in token_data]  # Extract only token counts

        # Sort distribution and track positions
        token_distribution = dict(sorted(Counter(token_counts_only).items()))
        min_tokens = min(token_counts_only)
        max_tokens = max(token_counts_only)
        mean_tokens = sum(token_counts_only) / len(token_counts_only)
        median_tokens = np.median(token_counts_only)
        median_tokens = float(median_tokens)

        # Save statistics to file
        f.write(f"\nStatistics for '{column}' across all splits:\n")
        f.write(f"  Min tokens: {min_tokens}, Max tokens: {max_tokens}\n")
        f.write(f"  Mean tokens: {mean_tokens:.2f}\n")
        f.write(f"  Median tokens: {median_tokens:.2f}\n")
        f.write(f"  Token distribution: {token_distribution}\n")

        # Save per-sample details
        f.write(f"\nPer-sample token counts for '{column}':\n")
        for token_count, split, idx in sorted(token_data, key=lambda x: x[0]):  # Sort by token count
            f.write(f"  Token Count: {token_count}, Split: {split}, Position: {idx}\n")

        print(f"Saved statistics for '{column}' across all splits to {stats_file}")

        # Plot and save histogram
        def format_number(x):
            return f'{x:,.0f}'


        plt.figure(figsize=(10, 6))

        plt.hist(token_counts_only, bins=90, edgecolor='black', alpha=0.7, color="skyblue")

        plt.axvline(mean_tokens, color='red', linestyle='dashed', linewidth=2, label=f"Mean: {format_number(mean_tokens)}")
        plt.axvline(median_tokens, color='green', linestyle='dashed', linewidth=2, label=f"Median: {format_number(median_tokens)}")

        plt.xticks(rotation=45)

        formatter = FuncFormatter(lambda x, _: f'{x:,.0f}')
        plt.gca().xaxis.set_major_formatter(formatter)
        plt.gca().yaxis.set_major_formatter(formatter)

        plt.xlabel("# Tokens")
        plt.ylabel("Frequency")
        plt.title(f"Token Count Distribution ({column})")
        plt.legend()
        plt.tight_layout()

        plt.savefig(f"token_distribution_cleaned_html_24k{column}_fancy.png")
        plt.close()

print(f"All statistics saved to {stats_file}")
