from collections import Counter
from datasets import load_from_disk

dataset = load_from_disk("/vol/tmp/stolzenp/training/split_dataset_cleaned_filtered_24k")

# Define output file
output_file = "language_statistics_24k.txt"

# Compute and save language statistics
with open(output_file, "w", encoding="utf-8") as f:
    for split in dataset.keys():
        lang_counts = Counter(dataset[split]["language"])  # Count occurrences
        f.write(f"Language distribution in {split}:\n")
        for lang, count in lang_counts.most_common():
            f.write(f"  {lang}: {count}\n")
        f.write("\n")

print(f"Results saved to {output_file}")
