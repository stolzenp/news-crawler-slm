import argparse
import json
from collections import Counter

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

from common.utils import get_args_from_config


def repetition_severity_ratio(text, n=5) -> float:
    """Calculate the ratio of repeated n-grams in a text (including how many times they repeat)."""

    # split text into tokens
    tokens = text.split()

    # if fewer tokens than desired n-grams return 0.0
    if len(tokens) < n:
        return 0.0

    # get n-grams and count for each of them
    ngrams = [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    total = len(ngrams)
    counts = Counter(ngrams)

    # get repetition ratio
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return repeated / total if total > 0 else 0.0


def exclude_degeneration_results(
    input_path, output_path, degen_path, mean_output_path, histogram_path, repetition_threshold=0.1, n_grams_size=5
):
    """Filters out degeneration results from metrics results and creates a histogram based on a repetition ratio."""

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)  # results are in JSON format

    filtered_data = []
    repetition_ratio_values = []
    degen_data = []
    degen_count = 0

    # check each result for degeneration
    for obj in data:
        output = obj.get("inference_output")
        repetition_ratio_value = repetition_severity_ratio(output, n_grams_size)
        obj["repetition_ratio"] = repetition_ratio_value
        repetition_ratio_values.append(repetition_ratio_value)

        if repetition_ratio_value < repetition_threshold:
            filtered_data.append(obj)
        else:
            degen_data.append(obj)
            degen_count += 1

    print(f"Filtered out {degen_count}  degenerated outputs.\n Remaining: {len(filtered_data)} samples\n")

    # prepare and plot histogram for repetition ratios
    ratio_mean = float(np.mean(repetition_ratio_values))
    ratio_median = float(np.median(repetition_ratio_values))

    plt.figure(figsize=(10, 6))

    plt.hist(repetition_ratio_values, bins="fd", edgecolor="black", alpha=0.7, color="skyblue")

    formatter = FuncFormatter(lambda x, _: f"{x:,.2f}")
    plt.axvline(ratio_mean, color="red", linestyle="dashed", linewidth=2, label=f"Mean: {formatter(ratio_mean)}")
    plt.axvline(
        ratio_median, color="green", linestyle="dashed", linewidth=2, label=f"Median: {formatter(ratio_median)}"
    )

    plt.xticks(rotation=45)

    plt.gca().xaxis.set_major_formatter(formatter)
    plt.gca().yaxis.set_major_formatter(formatter)

    plt.xlabel("Repetition Ratio")
    plt.ylabel("Frequency")
    plt.title(f"Repetition Ratio Distribution (n-grams={n_grams_size})")
    plt.legend()
    plt.tight_layout()

    plt.savefig(histogram_path)
    plt.close()

    print(f"Repetition ratio histogram saved to {histogram_path}\n")

    # save degeneration outputs separately for investigation
    if degen_data:
        with open(degen_path, "w", encoding="utf-8") as f:
            json.dump(degen_data, f, ensure_ascii=False, indent=4)

        print(f"Degeneration outputs saved to {degen_path}\n")

    # get all numerical attributes from remaining results
    numeric_attributes = [key for obj in filtered_data for key, value in obj.items() if isinstance(value, (int, float))]

    # calculate means for each numerical attribute
    if filtered_data:

        # save filtered results
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(filtered_data, f, ensure_ascii=False, indent=4)

        print(f"Filtered results with no degeneration saved to {output_path}\n")

        means = {attr: sum(obj[attr] for obj in filtered_data) / len(filtered_data) for attr in numeric_attributes}

        # save new mean values
        with open(mean_output_path, "w", encoding="utf-8") as f:
            print("Mean values for filtered results:\n")
            for metric, mean_value in means.items():
                message = f"{metric} mean: {mean_value}\n"
                f.write(message)
                print(message)

        print(f"New mean values saved to {mean_output_path}\n")

    else:
        print("No elements remain after filtering.")


if __name__ == "__main__":
    # get args from the config file
    eval_args = get_args_from_config("evaluation_settings")
    raw_metrics_file = eval_args["raw_metrics_file"]
    rep_threshold = eval_args["repetition_threshold"]
    n_grams = eval_args["n_grams"]
    output_dir = eval_args["base_output_dir"]

    # support arguments to enable quick setting change in CLI
    parser = argparse.ArgumentParser(description="Filter out degeneration results")
    parser.add_argument("-i", "--input", type=str, help="Metric scores file (.json)")
    parser.add_argument("-rep", "--rep-threshold", type=float, default=0.1, help="Max allowed repetition ratio")
    parser.add_argument("-n", "--n-gram", type=int, default=5, help="N-gram size for repetition detection")

    # assign arguments to variables
    args = parser.parse_args()

    # validate arguments
    if args.input is None and raw_metrics_file is None:
        print("No input file specified. Exiting.")
        exit()
    elif args.input is not None:
        raw_metrics_file = args.input

    if args.rep_threshold is None and rep_threshold is None:
        print("No repetition threshold specified. Exiting.")
        exit()
    elif args.rep_threshold:
        rep_threshold = args.rep_threshold

    if args.n_gram is None and n_grams is None:
        print("No n-gram size specified. Exiting.")
        exit()
    elif args.n_gram:
        n_grams = args.n_gram

    # set the input file path
    input_file = f"{output_dir}/{raw_metrics_file}"

    # set output paths
    output_file = f"{output_dir}/nodegen_{raw_metrics_file}"
    degen_file = f"{output_dir}/degen_{raw_metrics_file}"
    output_means_file = output_file.replace("results.json", "means.txt")
    plot_file = output_file.replace("results.json", f"repetition_hist_n{n_grams}.png")

    # filter out degeneration results
    exclude_degeneration_results(
        input_file, output_file, degen_file, output_means_file, plot_file, rep_threshold, n_grams
    )
