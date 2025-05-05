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

# prompt templates are defined in utils.py
from utils import get_args_from_config, format_prompts

def compute_perplexity(input_text, model_instance, tokenizer_instance):
    """Computes perplexity."""

    tokens = tokenizer_instance(input_text, return_tensors="pt").to("cuda")
    labels = tokens["input_ids"].clone()

    with torch.no_grad():
        outputs = model_instance(**tokens, labels=labels)
    perplexity = torch.exp(outputs.loss).item()

    return perplexity


def compute_inference_metrics(input_text, max_generation_length, gold_output, model_instance, tokenizer_instance):
    """Computes scores of inference metrics: Rouge-L, Levenshtein, Damerau, Jaro-Winkler Similarity."""

    tokens = tokenizer_instance(input_text, return_tensors="pt").to("cuda")

    with torch.no_grad():
        output_ids = model_instance.generate(**tokens, max_new_tokens=max_generation_length)

    output = tokenizer_instance.decode(output_ids[0][tokens["input_ids"].shape[1]:], skip_special_tokens=True)

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
    # get evaluation arguments from the config file
    eval_args = get_args_from_config("evaluation_settings")
    model_path = eval_args["model_name_or_path"]
    sequence_length = eval_args["sequence_length"]
    max_generation_length = eval_args["max_generation_length"]
    dataset_path = eval_args["dataset_path"]
    target_column = eval_args["target_column"]
    output_dir = eval_args["base_output_dir"]

    # add argument support for quick setting changes
    parser = argparse.ArgumentParser(description="Evaluation metrics")
    parser.add_argument("-t", "--target", choices=['plaintext', 'json'], help="Target: plaintext or json")
    parser.add_argument("-p", "--perplexity", action='store_true', help="Compute perplexity")
    parser.add_argument("-i", "--inference", action='store_true', help="Run inference metrics")
    parser.add_argument("-full", "--full-eval", action='store_true', help="Run all evaluation metrics")

    # assign arguments to variables
    args = parser.parse_args()
    target = args.target
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

    # raise an error if max_generation_length is too small
    if max_generation_length < 1:
        print("Invalid max_generation_length. Exiting.")
        return

    # load model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=sequence_length,
        dtype=None,
        load_in_4bit=True,
        device_map="cuda",
    )
    FastLanguageModel.for_inference(model)  # enable Unsloth's native 2x faster inference

    # load test dataset
    dataset = load_from_disk(dataset_path)
    test_set = dataset["test"]

    # get eos_token to prevent infinite generation loop
    eos_token = tokenizer.eos_token

    # add prompts for perplexity and inference metrics evaluation
    test_set = test_set.map(
        lambda examples: format_prompts(
            examples, 
            input_column="html", 
            output_column=target, 
            eos_token=eos_token, 
            for_training=False,
            compute_perplexity=args.perplexity,
            compute_inference=args.inference
        ), 
        batched=True
    )

    # create a list for results
    results = []

    # compute metrics for each test sample
    for index in tqdm(range(len(test_set)), total=len(test_set), desc="Processing samples"):
        sample = test_set[index]

        # create a result dictionary
        result_dict = {
            "html": sample["html"],
            f"{target}_gold": sample[target],
        }

        # compute perplexity
        if args.perplexity:
            perplexity = compute_perplexity(sample["text_ppl"], model_instance=model, tokenizer_instance=tokenizer)
            result_dict["Perplexity"] = perplexity

        # generate output and compute inference metrics
        if args.inference:
            inf_metrics = compute_inference_metrics(sample["text_inf"], max_generation_length, sample[target], model_instance=model, tokenizer_instance=tokenizer)
            # update result_dict with all metrics
            result_dict.update(inf_metrics)

        results.append(result_dict)

    filename_prefix = "all_metrics"
    if not args.perplexity:
        filename_prefix = "inference_metrics"
    elif not args.inference:
        filename_prefix = "perplexity"

    output_file = f"{output_dir}/{target}/{filename_prefix}_results.json"
    means_output_file = f"{output_dir}/{target}/{filename_prefix}_means.json"

    # save results to a JSON file
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
