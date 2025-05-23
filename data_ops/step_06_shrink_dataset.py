import os

from datasets import Dataset, DatasetDict, load_from_disk

from common.utils import get_args_from_config


def limit_samples_per_publisher(current_dataset, max_samples=50):
    """Limit the number of samples per publisher to a specified maximum."""

    grouped = {}

    for sample in current_dataset:
        publisher = sample["publisher"]
        if publisher not in grouped:
            grouped[publisher] = []
        if len(grouped[publisher]) < max_samples:
            grouped[publisher].append(sample)

    # flatten the grouped dictionary back into a list
    new_data = [sample for samples in grouped.values() for sample in samples]

    return Dataset.from_list(new_data)


if __name__ == "__main__":
    data_args = get_args_from_config("data_ops_settings")
    filtered_dataset_dir = data_args["filtered_dataset_directory"]
    shrinked_dataset_dir = data_args["shrinked_dataset_directory"]
    samples_per_publisher = data_args["samples_per_publisher"]

    # load dataset
    dataset = load_from_disk(filtered_dataset_dir)

    # print splits and sizes of dataset
    for split, data in dataset.items():
        print("Split sizes before limiting:\n")
        print(f"Split: {split}, Size: {len(data)}\n")

    # initialize dictionary for shrinked dictionary
    shrinked_dataset = {}

    # shrink every split except 'train'
    for split in dataset.keys():
        if split == "train":
            shrinked_dataset[split] = dataset[split]
            continue
        shrinked_dataset[split] = limit_samples_per_publisher(dataset[split], samples_per_publisher)

    # turn dictionary into dataset
    shrinked_dataset = DatasetDict(shrinked_dataset)

    # print splits and sizes of shrinked dataset
    for split, data in shrinked_dataset.items():
        print("Split sizes after limiting:\n")
        print(f"Split: {split}, Size: {len(data)}")

    # save shrinked dataset
    os.makedirs(shrinked_dataset_dir, exist_ok=True)
    shrinked_dataset.save_to_disk(shrinked_dataset_dir)
