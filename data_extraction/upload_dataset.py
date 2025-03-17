from datasets import load_from_disk

# Load dataset from local directory
dataset = load_from_disk("/vol/tmp/stolzenp/training/dataset")

# Define Hugging Face repo ID
repo_id = "stolzenp/fundus-93K"

# Push to Hugging Face Hub
dataset.push_to_hub(repo_id)

print(f"Dataset uploaded to https://huggingface.co/datasets/{repo_id}")
