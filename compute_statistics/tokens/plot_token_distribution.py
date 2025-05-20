import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from common.utils import get_args_from_config
from compute_statistics.tokens.count_samples_over_token_limit import get_token_counts_for_column

def format_number(x):
    """Format numbers for display purposes."""
    return f'{x:,.0f}'

def plot_token_distribution(token_counts, column_name, mean_value, median_value, output_file):
    """Plot and save histogram of the given token distribution."""
    plt.figure(figsize=(10, 6))

    plt.hist(token_counts, bins=90, edgecolor='black', alpha=0.7, color="skyblue")

    plt.axvline(mean, color='red', linestyle='dashed', linewidth=2, label=f"Mean: {format_number(mean_value)}")
    plt.axvline(median, color='green', linestyle='dashed', linewidth=2, label=f"Median: {format_number(median_value)}")

    plt.xticks(rotation=45)

    formatter = FuncFormatter(lambda x, _: f'{x:,.0f}')
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.gca().yaxis.set_major_formatter(formatter)

    plt.xlabel("# Tokens")
    plt.ylabel("Frequency")
    plt.title(f"Token Count Distribution ({column_name})")
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_file)
    plt.close()

if __name__ == "__main__":
    # get args from the config file
    stats_args = get_args_from_config("statistics_settings")
    output_dir = stats_args["output_dir"]
    stats_file = stats_args["token_stats_file"]
    columns = stats_args["columns"]

    # set path
    stats_file = f"{output_dir}/{stats_file}"

    for column in columns:
        print(f"Plotting histogram for '{column}'...")
        column_token_counts = get_token_counts_for_column(column, stats_file)

        # calculating mean and median
        mean = float(np.mean(column_token_counts))
        median = float(np.median(column_token_counts))

        # plotting histogram
        plot_file = f"{output_dir}/token_distribution_{column}.png"
        plot_token_distribution(column_token_counts, column, mean, median, plot_file)
        print(f"Saved histogram for '{column}' to {plot_file}")
