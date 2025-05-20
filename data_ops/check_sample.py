import os
import json
import argparse

from datasets import load_from_disk

from common.utils import get_args_from_config

def check_sample(dataset_directory, dataset_split, split_columns, index, output_directory=os.getcwd()):
    """Check a sample and save it to a file."""

    # get the sample and save it
    relevant_dataset = load_from_disk(dataset_directory)
    dataset_name = os.path.basename(os.path.normpath(dataset_directory))
    sample_file = f"{output_directory}/{dataset_name}/{dataset_split}/{index}.json"

    # get relevant sample columns
    sample = relevant_dataset[split_columns][index]
    filtered_sample = {key: sample[key] for key in split_columns if key in sample}

    # save to file and create directories if necessary
    os.makedirs(os.path.dirname(sample_file), exist_ok=True)
    with open(sample_file, "w", encoding="utf-8") as f:
        json.dump(filtered_sample, f, ensure_ascii=False, indent=4)

    print(json.dumps(filtered_sample, indent=2))
    print(f"Sample saved to {sample_file}")

if __name__ == "__main__":
    # get data ops arguments from the config file
    data_args = get_args_from_config("model_training_settings")
    dataset_dir = data_args["dataset_directory"]
    split = data_args["split"]
    sample_id = data_args["sample_id"]
    columns = data_args["columns"]
    output_dir = data_args["base_output_dir"]

    # support command line arguments for quick checks
    parser = argparse.ArgumentParser(description="Check sample in dataset")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Dataset directory (expects splits)")
    parser.add_argument("--split", type=str, required=True, help="Split to check (e.g., train, val, test)")
    parser.add_argument("--sample_id", type=int, required=True, help="Sample ID to check")
    parser.add_argument("--columns", nargs="+", type=str, help="Columns to check (all if not specified)")

    # parse command line arguments
    args = parser.parse_args()

    # validate dataset_dir argument and load dataset for checks
    if not args.dataset_dir and not dataset_dir:
        print("Must specify dataset_dir")
        exit(1)
    elif args.dataset_dir:
        dataset_dir = args.dataset_dir
    dataset = load_from_disk(dataset_dir)

    # validate split
    if not args.split and not split:
        print("No split specified")
        exit(1)
    elif args.split:
        split = args.split

    if split not in dataset:
        raise ValueError(f"Invalid split: {split}")

    # validate sample_id
    if not args.sample_id and not sample_id:
        print("No sample_id provided")
        exit(1)
    elif args.sample_id:
        sample_id = args.sample_id

    if sample_id < 0:
        raise ValueError("Sample ID must be a positive integer.")
    elif sample_id >= len(dataset[split]):
        raise ValueError("Sample ID is out of range.")

    # validate columns
    if not args.columns and not columns:
        columns = dataset[split].column_names
    else:
        if args.columns:
            columns = args.columns

        for column in columns:
            if column not in dataset[split].column_names:
                raise ValueError(f"Invalid column: {column}")

    # check the specified sample
    check_sample(dataset_dir, split, columns, output_dir)