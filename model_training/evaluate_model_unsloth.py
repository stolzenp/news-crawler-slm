import torch
import json
from unsloth import FastLanguageModel
from datasets import load_from_disk

# Load model and tokenizer
checkpoint_path = "/vol/tmp/stolzenp/training/qwen2.5-1.5b_24k+8k/results/plaintext/checkpoint-70458"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name= checkpoint_path,
    max_seq_length=32768,  # Adjust based on your dataset
    dtype=None,
    load_in_4bit=True,
    device_map="cuda",
)

dataset_path = "/vol/tmp/stolzenp/training/split_dataset_cleaned_filtered_24k"
dataset = load_from_disk(dataset_path)
test_set = dataset["test"]

# add prompt layout if bad results
def compute_perplexity_and_generate(input_text, target_text):
    """Computes perplexity and generates output."""
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.cuda()
    target_ids = tokenizer(target_text, return_tensors="pt").input_ids.cuda()

    with torch.no_grad():
        # Compute loss for perplexity
        outputs = model(input_ids, labels=target_ids)
        loss = outputs.loss
        perplexity = torch.exp(loss).item()

        # Generate text
        generated_ids = model.generate(input_ids, max_new_tokens=8192)  # Adjust max_new_tokens if needed
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return perplexity, generated_text


# Compute perplexity and output for each test sample
results = []
for sample in test_set:
    html = sample["html"]
    ground_truth = sample["plain_text"]

    ppl, model_output = compute_perplexity_and_generate(html, ground_truth)

    results.append({
        "html": html,
        "plain_text_gold": ground_truth,
        "plain_text_generate": model_output,
        "perplexity": ppl
    })

# Save results to JSON file
output_file = "/vol/tmp/stolzenp/training/qwen2.5-1.5b_24k+8k/results/plaintext/perplexity_results.json"
with open(output_file, "w") as f:
    json.dump(results, f, indent=4)

print(f"Perplexity results saved to {output_file}")

