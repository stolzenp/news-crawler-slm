import torch
import json
import numpy as np
from tqdm import tqdm
from unsloth import FastLanguageModel
from transformers import AutoTokenizer, AutoModel
from datasets import load_from_disk

# Load model and tokenizer
checkpoint_path = "/vol/tmp/stolzenp/training/ReaderLM-v2_24k+8k_cl_loss/results/plaintext/checkpoint-70458"
#tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
#model = AutoModel.from_pretrained(checkpoint_path)
print(checkpoint_path)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name= checkpoint_path,
    max_seq_length=32768,
    dtype=None,
    load_in_4bit=True,
    device_map="cuda",
)

print(model)

dataset_path = "/vol/tmp/stolzenp/training/shrinked_split_dataset_cleaned_filtered_24k"
dataset = load_from_disk(dataset_path)
test_set = dataset["test"]

def compute_perplexity(text):
    """Computes perplexity."""
    tokens = tokenizer(text, return_tensors="pt").to("cuda")
    labels = tokens["input_ids"].clone()

    with torch.no_grad():
        # Compute loss for perplexity
        outputs = model(**tokens, labels=labels)
        loss = outputs.loss
        perplexity = torch.exp(loss).item()

    return perplexity

prompt = """
Input:
{}

Output:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    inputs       = examples["html"]
    outputs      = examples["plain_text"]
    texts = []
    for input, output in zip(inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = prompt.format(input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

test_set = test_set.map(formatting_prompts_func, batched = True)

samples = test_set["text"]

# Compute perplexity for each test sample
results = []
perplexities = []
for index, sample in tqdm(enumerate(samples), total=len(samples), desc="Processing samples"):
    ppl = compute_perplexity(sample)

    perplexities.append(ppl)

    results.append({
        "html": test_set[index]["html"],
        "plain_text_gold": test_set[index]["plain_text"],
        "perplexity": ppl
    })

# Save results to JSON file
output_file = "/vol/tmp/stolzenp/results/ReaderLM-v2_finetuned_cl_loss/plaintext/perplexity_shrinked_results.json"
with open(output_file, "w") as f:
    json.dump(results, f, indent=4)

print(f"Perplexity results saved to {output_file}")

mean_perplexity = np.mean(perplexities)

with open('/vol/tmp/stolzenp/results/ReaderLM-v2_finetuned_cl_loss/plaintext/mean_perplexity_shrinked.txt', 'w') as f:
    f.write(f'Mean Perplexity: {mean_perplexity}\n')

print(f"Mean Perplexity: {mean_perplexity}")
