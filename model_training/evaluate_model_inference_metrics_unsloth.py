import torch
import json
import argparse
import numpy as np
from tqdm import tqdm
from unsloth import FastLanguageModel
from datasets import load_from_disk
import evaluate
import Levenshtein
from pyxdameraulevenshtein import damerau_levenshtein_distance
import jellyfish

parser = argparse.ArgumentParser(description="Inference metrics")
parser.add_argument("type", help="Input type")

args = parser.parse_args()
type = args.type
altered_type = args.type.replace("_", "")

# Load model and tokenizer
checkpoint_path = f"/vol/tmp/stolzenp/training/ReaderLM-v2_24k+8k_cl_loss/results/{altered_type}/checkpoint-70458"
#checkpoint_path = "jinaai/ReaderLM-v2"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name= checkpoint_path,
    max_seq_length=32768,
    dtype=None,
    load_in_4bit=True,
    device_map="cuda",
)

#model.to("cuda")

FastLanguageModel.for_inference(model) # Enable native 2x faster inference

dataset_path = "/vol/tmp/stolzenp/training/shrinked_split_dataset_cleaned_filtered_24k"
dataset = load_from_disk(dataset_path)
test_set = dataset["test"]
rouge = evaluate.load('rouge')

metric_scores = {
    "Rouge-L": [],
    "Levenshtein": [],
    "Damerau": [],
    "Jaro-Winkler": [],
}

results = []

prompt = """
Input:
{}

Output:
"""

def formatting_prompts_func(examples):
    inputs = examples["html"]
    texts = []
    for input in inputs:
        text = prompt.format(input)
        texts.append(text)
    return { "text" : texts, }

def compute_metrics(input_text, gold_output):
    """Computes evaluation metrics score."""
    tokens = tokenizer(input_text, return_tensors="pt").to("cuda")

    with torch.no_grad():
        # Generate output
        output_ids = model.generate(**tokens, max_new_tokens=8192)
        output = tokenizer.decode(output_ids[0][tokens["input_ids"].shape[1]:], skip_special_tokens=True)

        gold_output = json.dumps(gold_output)

        rouge_scores = rouge.compute(predictions=[output], references=[gold_output])
        rougeL = rouge_scores['rougeL']
        metric_scores["Rouge-L"].append(rougeL)

        levenshtein = Levenshtein.distance(output, gold_output)
        normalized_levenshtein = levenshtein/max(len(output), len(gold_output))
        metric_scores["Levenshtein"].append(normalized_levenshtein)

        damerau = damerau_levenshtein_distance(output, gold_output)
        metric_scores["Damerau"].append(damerau)

        jaro_winkler = jellyfish.jaro_winkler_similarity(output, gold_output)
        metric_scores["Jaro-Winkler"].append(jaro_winkler)

    return output

test_set = test_set.map(formatting_prompts_func, batched = True)

# Compute rouge and output for each test sample
for index, _ in tqdm(enumerate(test_set), total=len(test_set), desc="Processing samples"):
    output = compute_metrics(test_set[index]["text"], test_set[index][type])

    results.append({
        "html": test_set[index]["html"],
        f"{type}_gold": test_set[index][type],
        "output": output,
        "Rouge-L": metric_scores["Rouge-L"][index],
        "Levenshtein": metric_scores["Levenshtein"][index],
        "Damerau": metric_scores["Damerau"][index],
        "Jaro-Winkler": metric_scores["Jaro-Winkler"][index],
    })

# Save results to JSON file
output_file = f"/vol/tmp/stolzenp/results/ReaderLM-v2_finetuned_cl_loss/{altered_type}/inference_metrics_shrinked_results.json"
with open(output_file, "w") as f:
    json.dump(results, f, indent=4)

print(f"Inference metrics results saved to {output_file}")

with open(f'/vol/tmp/stolzenp/results/ReaderLM-v2_finetuned_cl_loss/{altered_type}/inference_metrics_means_shrinked.txt', 'w') as f:
    for metric_name, scores in metric_scores.items():
        mean = np.mean(scores)
        message = f"{metric_name} mean: {mean}\n"
        f.write(message)
        print(message)
