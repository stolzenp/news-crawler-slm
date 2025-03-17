from datasets import load_from_disk

dataset = load_from_disk("/vol/tmp/stolzenp/training/split_dataset_cleaned_filtered")

for split in dataset.keys():
    print(f"{split}: {dataset[split].num_rows} samples")