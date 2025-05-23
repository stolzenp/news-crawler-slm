import json
import argparse
import os

from common.utils import get_args_from_config

def exclude_degeneration_results(input_path, output_path, mean_output_path):
    """Filters out degeneration results from metrics results."""

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f) # results are in JSON format

    # all observed Damerau above 10000 were degenerations, so we exclude these samples
    filtered_data = [obj for obj in data if obj.get("Damerau") < 10000]

    # get all numerical attributes from remaining results
    numeric_attributes = [key for obj in filtered_data for key, value in obj.items() if isinstance(value, (int, float))]

    # calculate means for each numerical attribute
    if filtered_data:

        # save filtered results
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(filtered_data, f, ensure_ascii=False, indent=4)

        print(f"No degeneration results saved to {output_file}\n")

        means = {
            attr: sum(obj[attr] for obj in filtered_data) / len(filtered_data)
            for attr in numeric_attributes
        }

        # save new mean values
        with open(output_path, "w", encoding="utf-8") as f:
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
    target = eval_args["target_column"]
    output_dir = eval_args["base_output_dir"]

    # support arguments to enable quick setting change in CLI
    parser = argparse.ArgumentParser(description="Filter out degeneration results")
    parser.add_argument( "-i", "--input", type=str, help="Metric scores file (.json)")

    # assign arguments to variables
    args = parser.parse_args()

    # validate arguments
    if args.input is None and raw_metrics_file is None:
        print("No input file specified. Exiting.")
        exit()
    elif args.input is not None:
        raw_metrics_file = args.input

    if args.target is None and target is None:
        print("No target column specified. Exiting.")
        exit()
    if args.target is not None:
        target = args.target

    # set the input file path
    input_file = f"{output_dir}/{raw_metrics_file}"

    # set output paths
    prefix = "nodegen"
    output_file = f"{output_dir}/{prefix}_{raw_metrics_file}"
    output_means_file = output_file.replace('results.json', 'means.txt')

    # filter out degeneration results
    exclude_degeneration_results(input_file, output_file, output_means_file)

