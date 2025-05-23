import json
import os
from collections import Counter

import numpy as np
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoTokenizer

from common.utils import get_args_from_config
from compute_statistics.tokens.plot_token_distribution import plot_token_distribution


def get_token_counts(samples, tokenizer):
    """Tokenize and count tokens for a list of samples."""

    return [len(tokenizer.tokenize(entry)) for entry in tqdm(samples, desc="Tokenizing", ncols=80)]


def get_token_statistics(dataset, model_tokenizer, columns_to_process, output_file):
    """Compute and save token statistics on specific columns for a given dataset and tokenizer."""

    column_token_counts = {column: [] for column in columns_to_process}

    with open(output_file, "w") as f:
        for split in dataset.keys():
            print(f"Processing {split} split...")

            for column in columns_to_process:
                print(f"Processing column: {column}")

                # extract texts (handling JSON conversion if needed)
                if column == "json":
                    texts = [json.dumps(entry) for entry in dataset[split][column]]
                else:
                    texts = dataset[split][column]

                # tokenize and store token counts with metadata (split and position)
                token_counts = get_token_counts(texts, model_tokenizer)
                column_token_counts[column].extend(
                    (token_count, split, idx) for idx, token_count in enumerate(token_counts)
                )

        # compute statistics per column across all splits
        for column, token_data in column_token_counts.items():
            token_counts_only = [t[0] for t in token_data]  # extract only token counts

            # sort distribution and calculate statistics
            token_distribution = dict(sorted(Counter(token_counts_only).items()))
            min_tokens = min(token_counts_only)
            max_tokens = max(token_counts_only)
            mean_tokens = sum(token_counts_only) / len(token_counts_only)
            median_tokens = np.median(token_counts_only)
            median_tokens = float(median_tokens)

            # save overall statistics
            f.write(f"\nStatistics for '{column}' across all splits:\n")
            f.write(f"  Min tokens: {min_tokens}, Max tokens: {max_tokens}\n")
            f.write(f"  Mean tokens: {mean_tokens:.2f}\n")
            f.write(f"  Median tokens: {median_tokens:.2f}\n")
            f.write(f"  Token distribution: {token_distribution}\n")

            # save per-sample details
            f.write(f"\nPer-sample token counts for '{column}':\n")
            for token_count, split, idx in sorted(token_data, key=lambda x: x[0]):  # sort by token count
                f.write(f"  Token Count: {token_count}, Split: {split}, Position: {idx}\n")

            print(f"Saved statistics for '{column}' across all splits to {stats_file}")

            # plot and save histogram for the current column
            plot_file = f"{output_dir}/token_distribution_{column}.png"
            plot_token_distribution(token_counts_only, column, mean_tokens, median_tokens, plot_file)
            print(f"Saved histogram for '{column}' to {plot_file}")

    print(f"All statistics saved to {stats_file}")


if __name__ == "__main__":
    # get config arguments
    model_args = get_args_from_config("model_training_settings")
    stats_args = get_args_from_config("statistics_settings")

    # assign relevant arguments
    model_name = model_args["model_name_or_path"]
    dataset_dir = stats_args["dataset_directory"]
    stats_file = stats_args["token_stats_file"]
    output_dir = stats_args["output_dir"]
    columns = stats_args["columns"]

    # set the file path and create directories if needed
    stats_file = f"{output_dir}/{stats_file}"
    os.makedirs(os.path.dirname(stats_file), exist_ok=True)

    # load dataset and tokenizer
    chosen_dataset = load_from_disk(dataset_dir)
    chosen_tokenizer = AutoTokenizer.from_pretrained(model_name)

    # compute and save token statistics and plot histograms for each column
    get_token_statistics(chosen_dataset, chosen_tokenizer, columns, stats_file)
