import json
import argparse
import numpy as np
from tqdm import tqdm

import torch
from datasets import load_from_disk
import evaluate
import Levenshtein
from pyxdameraulevenshtein import damerau_levenshtein_distance
import jellyfish
from unsloth import FastLanguageModel

from utils import get_args_from_config

# define prompts for perplexity and inference metrics evaluation
PPL_PROMPT = "Input:\n{}\n\nOutput:\n{}"
INF_PROMPT = "Input:\n{}\n\nOutput:\n"

def formatting_prompts(examples, target_column, eos_token):
    inputs = examples["html"]
    outputs = examples[target_column]
    return {
        # prompts for perplexity evaluation (input and output)
        "text_ppl": [PPL_PROMPT.format(i, o) + eos_token for i, o in zip(inputs, outputs)],
        # prompts for inference (only input)
        "text_inf": [INF_PROMPT.format(i) for i in inputs],
    }


def compute_perplexity(input_text, model, tokenizer):
    """Computes perplexity."""
    tokens = tokenizer(input_text, return_tensors="pt").to("cuda")
    labels = tokens["input_ids"].clone()

    with torch.no_grad():
        outputs = model(**tokens, labels=labels)
    perplexity = torch.exp(outputs.loss).item()

    return perplexity


def compute_inference_metrics(input_text, gold_output, model, tokenizer):
    """Computes evaluation metrics score."""

    tokens = tokenizer(input_text, return_tensors="pt").to("cuda")

    with torch.no_grad():
        output_ids = model.generate(**tokens, max_new_tokens=8192)

    output = tokenizer.decode(output_ids[0][tokens["input_ids"].shape[1]:], skip_special_tokens=True)

    gold_output = json.dumps(gold_output)

    rouge = evaluate.load('rouge')
    rouge_l = rouge.compute(predictions=[output], references=[gold_output])['rougeL']

    levenshtein = Levenshtein.distance(output, gold_output)
    normalized_levenshtein = levenshtein / max(len(output), len(gold_output))

    damerau = damerau_levenshtein_distance(output, gold_output)

    jaro_winkler = jellyfish.jaro_winkler_similarity(output, gold_output)

    inf_metrics = {
        "inference_output": output,
        "Rouge-L": rouge_l,
        "Levenshtein": normalized_levenshtein,
        "Damerau": damerau,
        "Jaro-Winkler": jaro_winkler
    }

    return inf_metrics

def main():
    # get evaluation arguments from config file
    eval_args = get_args_from_config("evaluation_settings")
    model_path = eval_args["model_path"]
    sequence_length = eval_args["sequence_length"]
    dataset_path = eval_args["dataset_path"]
    target_column = eval_args["target_column"]
    output_dir = eval_args["output_dir"]

    # add parser arguments for quick
    parser = argparse.ArgumentParser(description="Evaluation metrics")
    parser.add_argument("-t", "--target", choices=['plaintext', 'json'], help="Target: plaintext or json")
    parser.add_argument("-p", "--perplexity", action='store_true', help="Compute perplexity")
    parser.add_argument("-i", "--inference", action='store_true', help="Run inference metrics")
    parser.add_argument("-full", "--full-eval", action='store_true', help="Run all evaluation metrics")

    # assign arguments to variables
    args = parser.parse_args()
    target = args.target
    altered_type = args.target.replace("_", "")
    if args.full_eval:
        args.perplexity = args.inference = True
    elif not args.perplexity and not args.inference:
        print("No evaluation metrics selected. Exiting.")
        return

    if not args.target and not target_column:
        print("No target column specified. Exiting.")
        return
    else:
        if args.target and target_column:
            print("Target column specified in config and argument. Using argument value.")
        target = target_column

    # load model and tokenizer
    # checkpoint_path = f"/vol/tmp/stolzenp/training/ReaderLM-v2_24k+8k_cl_loss/results/{altered_type}/checkpoint-70458"
    # checkpoint_path = "jinaai/ReaderLM-v2"
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=sequence_length,
        dtype=None,
        load_in_4bit=True,
        device_map="cuda",
    )
    FastLanguageModel.for_inference(model)  # enable Unsloth's native 2x faster inference

    # load test dataset
    #dataset_path = "/vol/tmp/stolzenp/training/shrinked_split_dataset_cleaned_filtered_24k"
    dataset = load_from_disk(dataset_path)
    test_set = dataset["test"]

    # get eos_token to prevent infinite generation loop
    eos_token = tokenizer.eos_token

    # add prompts for perplexity and inference metrics evaluation
    test_set = test_set.map(lambda examples: formatting_prompts(examples, target, eos_token), batched=True)

    # create results list
    results = []

    # compute metrics for each test sample
    for index in tqdm(range(len(test_set)), total=len(test_set), desc="Processing samples"):

        sample = test_set[index]

        # create result dictionary
        result_dict = {
            "html": sample["html"],
            f"{target}_gold": sample[target],
        }

        # compute perplexity
        if args.perplexity:
            perplexity = compute_perplexity(sample["text_ppl"], model, tokenizer)
            result_dict["Perplexity"] = perplexity

        # generate output and compute inference metrics
        if args.inference:
            inf_metrics = compute_inference_metrics(sample["text_inf"], sample[target], model, tokenizer)
            # update result_dict with all metrics
            result_dict.update(inf_metrics)

        results.append(result_dict)

    # f"/vol/tmp/stolzenp/results/ReaderLM-v2_finetuned_cl_loss/{altered_type}/inference_metrics_shrinked_results.json"
    # f"/vol/tmp/stolzenp/results/ReaderLM-v2_finetuned_cl_loss/{altered_type}/inference_metrics_means_shrinked.txt"

    filename_prefix = "all_metrics"
    if not args.perplexity:
        filename_prefix = "inference_metrics"
    else:
        filename_prefix = "perplexity"

    output_file = f"{output_dir}/{altered_type}/{filename_prefix}_shrinked_results.json"
    means_output_file = f"{output_dir}/{altered_type}/{filename_prefix}_means_shrinked.json"

    # save results to JSON file
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Inference metrics results saved to {output_file}")

    # calculate and save mean metrics
    with open(means_output_file, 'w') as f:
        # get all metric names from results
        metric_names = set()
        for result in results:
            metric_names.update(result.keys())

        # remove non-metric fields
        non_metrics = ["html", f"{target}_gold", "inference_output"]
        metric_names = [name for name in metric_names if name not in non_metrics]

        # calculate means for each metric
        for metric_name in metric_names:
            values = [result[metric_name] for result in results]
            mean = np.mean(values)
            message = f"{metric_name} mean: {mean}\n"
            f.write(message)
            print(message)

    print(f"Mean values saved to {means_output_file}")

if __name__ == "__main__":
    main()
