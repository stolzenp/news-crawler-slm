from unsloth import FastLanguageModel # repositioned so unsloth does not complain
import os
import ast
import json
import argparse
import numpy as np
from tqdm import tqdm

import torch
import wandb
from datasets import load_from_disk
import evaluate
import Levenshtein
from pyxdameraulevenshtein import damerau_levenshtein_distance
import jellyfish

# prompt templates are defined in utils.py
from common.utils import get_args_from_config, format_prompts

def compute_perplexity(input_text, model_instance, tokenizer_instance):
    """Computes perplexity."""

    tokens = tokenizer_instance(input_text, return_tensors="pt").to("cuda")
    labels = tokens["input_ids"].clone()

    with torch.no_grad():
        outputs = model_instance(**tokens, labels=labels)
    perplexity = torch.exp(outputs.loss).item()

    return perplexity

def safe_transform_to_json(pred_str):
    """Checks if the predicted string is valid JSON and returns the parsed dictionary if valid."""

    try:
        return ast.literal_eval(pred_str)
    except (SyntaxError, ValueError):
        return None

def get_key_sets(pred_keys, gold_keys):
    """Returns sets of extra keys, missing keys and common keys between two key sets."""

    extra = pred_keys - gold_keys
    missing = gold_keys - pred_keys
    common = pred_keys & gold_keys

    return extra, missing, common

def collect_unique_keys_and_types(data, prefix="", result=None):
    """Collects unique keys and types from a nested dictionary."""

    if result is None:
        result = {}

    if isinstance(data, dict):
        for key, value in data.items():
            new_prefix = f"{prefix}.{key}" if prefix else key
            if new_prefix not in result:
                result[new_prefix] = set()
            result[new_prefix].add(type(value).__name__)

            # recurse into value
            result |= collect_unique_keys_and_types(value, new_prefix, result)

    elif isinstance(data, list):
        for item in data:
            # only recurse if it is a dict or list
            if isinstance(item, (dict, list)):
                result |= collect_unique_keys_and_types(item, prefix, result)

    return result

def extract_all_text(data):
    """Extracts all text from a nested dictionary."""

    texts = []

    if isinstance(data, str):
        texts.append(data)
    elif isinstance(data, dict):
        for value in data.values():
            texts.append(extract_all_text(value))
    elif isinstance(data, list):
        for item in data:
            texts.append(extract_all_text(item))

    return " ".join(t for t in texts if t)  # join strings

def calculate_text_similarity_metrics(gold_text, pred_text):
    """Computes text similarity metrics: Rouge-L, BLEU, METEOR, Levenshtein, Damerau, Jaro-Winkler Similarity."""



    rouge = evaluate.load('rouge')
    rouge_l = rouge.compute(predictions=[pred_text], references=[gold_text])['rougeL']

    bleu = evaluate.load('bleu')
    bleu_score = bleu.compute(predictions=[pred_text], references=[gold_text])['bleu']

    meteor = evaluate.load('meteor')
    meteor_score = meteor.compute(predictions=[pred_text], references=[gold_text])['meteor']

    levenshtein = Levenshtein.distance(pred_text, gold_text)
    normalized_levenshtein = levenshtein / max(len(pred_text), len(gold_text))

    damerau = damerau_levenshtein_distance(pred_text, gold_text)

    jaro_winkler = jellyfish.jaro_winkler_similarity(pred_text, gold_text)

    return {
        "Rouge-L": rouge_l,
        "BLEU": bleu_score,
        "METEOR": meteor_score,
        "Levenshtein": normalized_levenshtein,
        "Damerau": damerau,
        "Jaro-Winkler": jaro_winkler
    }


def evaluate_json(prediction, gold_data):
    """Validates JSON structure, collects True Positives (TP), False Positives (FP) and False Negatives (FN) and computes text similarity metrics for body field."""

    # check if the predicted JSON serializable dictionary is valid
    valid_json = safe_transform_to_json(prediction)

    # stop extra evaluation if JSON is invalid
    if valid_json is None:
        return {
            "valid_json": 0
        }

    # we exclude True Negatives (TN) in our scenario since usually many fields are None
    field_scores = {
        "valid_json": 1,
        "TP": 0,
        "FP": 0,
        "FN": 0,
    }

    # get keys
    gold_keys = set(gold_data)
    pred_keys = set(valid_json)

    # get extra, missing and common keys
    extra_keys, missing_keys, common_keys = get_key_sets(pred_keys, gold_keys)

    # punish extra keys in prediction
    field_scores["FP"] += len(extra_keys)

    # punish if prediction misses key
    field_scores["FN"] += len(missing_keys)

    for key in common_keys:
        # get values
        pred_value = valid_json[key]
        gold_value = gold_data[key]

        # handle special case body field (only dict in all gold data)
        if key == "body":
            # since text generation is never perfect, we only check unique key overlap and value type match
            gold_body_data = collect_unique_keys_and_types(gold_value)
            pred_body_data = collect_unique_keys_and_types(pred_value)

            # get subkeys
            gold_subkeys = set(gold_body_data)
            pred_subkeys = set(pred_body_data)

            # get extra, missing and common subkeys
            extra_subkeys, missing_subkeys, common_subkeys = get_key_sets(pred_subkeys, gold_subkeys)

            # punish unexpected subfield
            for _ in extra_subkeys:
                field_scores["FP"] += 1

            # punish missing subfield
            for _ in missing_subkeys:
                field_scores["FN"] += 1

            for subkey in common_subkeys:
                # punish if types are not equal (missed correct type (FN) and predicted wrong type (FP))
                if gold_body_data[subkey] != pred_body_data[subkey]:
                    field_scores["FN"] += 1
                    field_scores["FP"] += 1

            # extract text from the body field
            gold_body_text = extract_all_text(gold_value)
            pred_body_text = extract_all_text(pred_value)

            # calculate text similarity metrics
            body_metrics = calculate_text_similarity_metrics(gold_body_text, pred_body_text)
            body_metrics = {f"body_{key}": value for key, value in body_metrics.items()}
            field_scores.update(body_metrics)

        # punish if types do not match (missed correct type (FN) and predicted wrong type (FP))
        elif type(gold_value) != type(pred_value):
            field_scores["FN"] += 1
            field_scores["FP"] += 1

        # evaluate values for the list type (order in the list and duplicates do not matter)
        elif isinstance(gold_value, list) and isinstance(pred_value, list):
            # punish if lists do not have the same elements (missed correct value (FN) and predicted wrong value (FP))
            if set(gold_value) != set(pred_value):
                field_scores["FN"] += 1
                field_scores["FP"] += 1
            else:
                field_scores["TP"] += 1

        # punish hallucination
        elif gold_value is None and pred_value is not None:
            field_scores["FP"] += 1

        # punish incorrect value (missed correct value (FN) and predicted wrong value (FP))
        elif pred_value != gold_value:
            field_scores["FN"] += 1
            field_scores["FP"] += 1

        # values are identical
        elif pred_value == gold_value:
            field_scores["TP"] += 1

    return field_scores

def compute_final_json_metrics(results, sample_amount):
    """Computes final JSON evaluation metrics: Precision, Recall, F1 Score, Valid-JSON rate."""

    precision = results["TP"] / (results["TP"] + results["FP"])
    recall = results["TP"] / (results["TP"] + results["FN"])
    f1_score = 2 * precision * recall / (precision + recall)
    valid_json_rate = results["valid_json"] / sample_amount

    return {
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1_score,
        "Valid-JSON Rate": valid_json_rate
    }

def compute_inference_metrics(input_text, max_generation_length, target_column, gold_output, model_instance, tokenizer_instance):
    """Computes scores of inference metrics: Rouge-L, BLEU, METEOR, Levenshtein, Damerau, Jaro-Winkler Similarity. If the target column is JSON, it also computes JSON evaluation metrics."""

    tokens = tokenizer_instance(input_text, return_tensors="pt").to("cuda")

    with torch.no_grad():
        output_ids = model_instance.generate(**tokens, max_new_tokens=max_generation_length)

    output = tokenizer_instance.decode(output_ids[0][tokens["input_ids"].shape[1]:], skip_special_tokens=True)

    # handle extra JSON evaluation
    json_specific_metrics = {}
    if target_column == "json":
        json_specific_metrics = evaluate_json(output, gold_output)

        # turn gold output into string for text similarity metrics
        gold_output = json.dumps(gold_output)

    text_similarity_metrics = calculate_text_similarity_metrics(gold_output, output)

    # turn output into JSON again for the result
    if json_specific_metrics.get("valid_json", 0) == 1:
        output = safe_transform_to_json(output)

    inf_metrics = {
        "inference_output": output,
        **text_similarity_metrics,
        **json_specific_metrics,
    }

    return inf_metrics

def evaluate_model(
        dataset,
        model,
        tokenizer,
        target_column,
        max_generation_length,
        compute_ppl=False,
        compute_inf=False,
        log_to_wandb=False,
        wandb_step=0,
):
    """Evaluates a model on a dataset and saves results to a JSON file and mean scores to a text file."""

    assert compute_ppl or compute_inf, "At least one of compute_ppl or compute_inf must be True."

    # get eos_token to prevent infinite generation loop
    eos_token = tokenizer.eos_token

    # add prompts for perplexity and inference metrics evaluation
    dataset = dataset.map(
        lambda examples: format_prompts(
            examples,
            input_column="html",
            output_column=target_column,
            eos_token=eos_token,
            for_training=False,
            compute_perplexity=compute_ppl,
            compute_inference=compute_inf
        ),
        batched=True
    )

    # create a list for results
    results = []

    # keep track of JSON-specific metrics
    aggregated_json_metrics = {"TP": 0, "FP": 0, "FN": 0, "valid_json": 0}
    body_metric_sums = {}
    valid_json_count = 0

    # compute metrics for each test sample
    for sample in tqdm(dataset, desc="Evaluating"):
        # create a result dictionary
        result = {
            "html": sample["html"],
            f"{target_column}_gold": sample[target_column]
        }

        # compute perplexity
        if compute_ppl:
            ppl = compute_perplexity(
                sample["text_ppl"], model_instance=model, tokenizer_instance=tokenizer
            )
            result["Perplexity"] = ppl

        # generate output and compute inference metrics
        if compute_inf:
            inf_metrics = compute_inference_metrics(
                sample["text_inf"], max_generation_length, target_column,
                sample[target_column], model_instance=model, tokenizer_instance=tokenizer
            )
            # update result_dict with all metrics
            result.update(inf_metrics)

            if target_column == "json":
                # accumulate JSON metrics for final scores
                for key in aggregated_json_metrics:
                    aggregated_json_metrics[key] += inf_metrics.get(key, 0)

                # accumulate body_* metrics only for valid JSON
                if inf_metrics.get("valid_json", 0) == 1:
                    valid_json_count += 1
                    for k, v in inf_metrics.items():
                        if k.startswith("body_") and isinstance(v, (int, float)):
                            body_metric_sums[k] = body_metric_sums.get(k, 0) + v

        results.append(result)

    # get all metric names from results
    metric_names = set()
    for result in results:
        metric_names.update(result.keys())

    # exclude non-metrics and aggregated fields
    non_metrics = {"html", f"{target_column}_gold", "inference_output", "TP", "FP", "FN", "valid_json"}
    metric_names = [name for name in metric_names if name not in non_metrics]

    mean_metrics = {}

    # calculate means for each metric
    for metric in metric_names:
        values = [r[metric] for r in results if metric in r]
        if metric.startswith("body_") and valid_json_count > 0:
            mean = body_metric_sums[metric] / valid_json_count
        else:
            mean = np.mean(values) if values else 0.0
        mean_metrics[metric] = mean

    if target_column == "json":
        # get final aggregated JSON metrics and add them to mean metrics
        final_json_metrics = compute_final_json_metrics(aggregated_json_metrics, sample_amount=len(dataset))
        mean_metrics.update(final_json_metrics)

    if log_to_wandb:
        wandb.log(mean_metrics, step=wandb_step)

    return results, mean_metrics

def main():
    # get evaluation arguments from the config file
    eval_args = get_args_from_config("evaluation_settings")
    model_path = eval_args["model_name_or_path"]
    sequence_length = eval_args["sequence_length"]
    max_generation_length = eval_args["max_generation_length"]
    dataset_path = eval_args["dataset_path"]
    target = eval_args["target_column"]
    run_all_metrics = eval_args["run_all_metrics"]
    output_dir = eval_args["base_output_dir"]
    split = eval_args["split"]

    # get wandb arguments
    wandb_project = eval_args["wandb_project"]
    wandb_entity = eval_args["wandb_entity"]
    wandb_run_name = eval_args["wandb_run_name"]
    wandb_flag = eval_args["log_to_wandb"]

    # add argument support for quick setting changes
    parser = argparse.ArgumentParser(description="Evaluation metrics")
    parser.add_argument("-t", "--target", choices=['plaintext', 'json'], help="Target: plaintext or json")
    parser.add_argument("-p", "--perplexity", action='store_true', help="Compute perplexity")
    parser.add_argument("-i", "--inference", action='store_true', help="Run inference metrics")
    parser.add_argument("-full", "--full-eval", action='store_true', help="Run all evaluation metrics")

    # assign arguments to variables
    args = parser.parse_args()

    if args.full_eval or run_all_metrics:
        args.perplexity = args.inference = True
    elif not args.perplexity and not args.inference:
        print("No evaluation metrics selected. Exiting.")
        return

    if not args.target and not target:
        print("No target column specified. Exiting.")
        return
    elif args.target:
        print("Using argument value for target column.")
        target = args.target

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
    splits = dataset.keys()
    if split and split in splits:
        dataset = dataset[split]
    else:
        print(f"Split '{split}' not found in the dataset. Using the default split.")

    wandb.init(project=wandb_project, entity=wandb_entity, name=wandb_run_name)

    results, mean_metrics = evaluate_model(dataset, model, tokenizer, target, max_generation_length, args.perplexity, args.inference, wandb_flag)

    filename_prefix = "all_metrics"
    if not args.perplexity:
        filename_prefix = "inference_metrics"
    elif not args.inference:
        filename_prefix = "perplexity"

    # expand the output path
    output_dir = f"{output_dir}/{dataset_path.split('/')[-1]}"
    if split and split in splits:
        output_dir = f"{output_dir}/{split}"

    # create directories if needed
    os.makedirs(output_dir, exist_ok=True)

    output_file = f"{output_dir}/{filename_prefix}_results.json"
    means_output_file = f"{output_dir}/{filename_prefix}_means.txt"

    # save results to a JSON file
    with open(output_file, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"Inference metrics results saved to {output_file}")

    # save means to a text file
    with open(means_output_file, "w") as f:
        for metric_name, mean in mean_metrics.items():
            message = f"{metric_name}: {mean}\n"
            f.write(message)
            print(message)
    print(f"Mean values saved to {means_output_file}")

if __name__ == "__main__":
    main()
