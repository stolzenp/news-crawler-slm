from lxml.parser import result
from unsloth import FastLanguageModel, is_bfloat16_supported # repositioned so unsloth does not complain
from datetime import datetime
import argparse
import math
import json
import os

import wandb
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

# using custom hf code to enable contrastive loss and inference metrics evaluation
from model_training.custom_hf_instances import CustomTrainer, custom_from_pretrained
from evaluation.evaluate_model import compute_inference_metrics, compute_final_json_metrics
from common.utils import format_prompts, get_args_from_config

model_args = get_args_from_config("model_training_settings")

# dataset and model settings
dataset_dir = model_args["split_dataset_directory"]
input_column = model_args["input_column"]
target = model_args["target_column"]
model_name = model_args["model_name_or_path"]
output_base_dir = model_args["output_base_dir"]
use_cl_loss = model_args["use_contrastive_loss"]
sequence_length = model_args["sequence_length"]
max_generation_length = model_args["max_generation_length"]
random_seed = model_args["seed"]
load_in_4bit = model_args["load_in_4bit"]

# peft settings
peft_r = model_args["peft_r"]
peft_target_modules = model_args["peft_target_modules"]
peft_lora_alpha = model_args["peft_lora_alpha"]
peft_lora_dropout = model_args["peft_lora_dropout"]
peft_bias = model_args["peft_bias"]
peft_use_gradient_checkpointing = model_args["peft_use_gradient_checkpointing"]
peft_loftq_config = model_args["peft_loftq_config"]

# trainer settings
dataset_num_proc = model_args["dataset_num_proc"]
warmup_steps = model_args["warmup_steps"]
learning_rate = model_args["learning_rate"]
lr_scheduler_type = model_args["learning_rate_scheduler_type"]
training_epochs = model_args["training_epochs"]
per_device_train_batch_size = model_args["per_device_train_batch_size"]
gradient_accumulation_steps = model_args["gradient_accumulation_steps"]
optimizer = model_args["optimizer"]
weight_decay = model_args["weight_decay"]
eval_strategy = model_args["eval_strategy"]
eval_steps = model_args["eval_steps"]
per_device_eval_batch_size = model_args["per_device_eval_batch_size"]
batch_eval_metrics = model_args["batch_eval_metrics"]
eval_on_start = model_args["eval_on_start"]
include_for_metrics = model_args["include_for_metrics"]
save_strategy = model_args["save_strategy"]
save_steps = model_args["save_steps"]
logging_steps = model_args["logging_steps"]
report_to = model_args["report_to"]
load_best_model_at_end = model_args["load_best_model_at_end"]
metric_for_best_model = model_args["metric_for_best_model"]
greater_is_better = model_args["greater_is_better"]

# wandb settings
wandb_project = model_args["wandb_project"]
wandb_entity = model_args["wandb_entity"]
wandb_run_name = model_args["wandb_run_name"]
wandb_run_ids_file = f"{output_base_dir}/wandb_run_ids.json"

# add argument support for quick setting changes
parser = argparse.ArgumentParser(description="Finetune model")
parser.add_argument("-t", "--target", choices=['plaintext', 'json'], help="Target: plaintext or json")
parser.add_argument("-cl", "--contrastive-loss", action='store_true', help="Run training with contrastive loss")
parser.add_argument("-r", "--resume", type=str, default=None, help="Weights & Biases run name for resuming run")

# parse arguments
args = parser.parse_args()

# create the output base directory if it does not exist
os.makedirs(output_base_dir, exist_ok=True)

# overwrite config value if argument passed
if args.contrastive_loss:
    use_cl_loss = True
if args.target:
    target = args.target

# apply the custom patch if contrastive loss is enabled
if use_cl_loss:
    AutoModelForCausalLM.from_pretrained = custom_from_pretrained

# get run id from wandb_run_ids.json if resume is requested
run_id = None
run_name = args.resume if args.resume else wandb_run_name
if run_name:
    with open(wandb_run_ids_file, "r") as f:
        run_ids = json.load(f)
        if run_name in run_ids:
            run_id = run_ids[run_name]
        else:
            print(f"Run name {run_name} not found in {wandb_run_ids_file}. Exiting.")
            exit(1)


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    # Unsloth supports RoPE Scaling, so any is possible if resources allow it
    max_seq_length = sequence_length, # we chose 32768 based on our resources
    dtype = None, # None for auto-detection
    load_in_4bit = load_in_4bit, # 4bit quantization for reducing memory usage; we set True
)

model = FastLanguageModel.get_peft_model(
    model,
    # our LoRA settings result in approximately 1-2% trainable parameters of total parameter count
    r = peft_r, # we chose 16
    target_modules = peft_target_modules, # we chose ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lora_alpha = peft_lora_alpha, # we chose 16
    lora_dropout = peft_lora_dropout, # we set 0 since Unsloth is optimized for this value
    bias = peft_bias, # we set "none" since Unsloth is optimized for this value
    use_gradient_checkpointing = peft_use_gradient_checkpointing, # we set "unsloth" which supports a very long context
    random_state = random_seed, # we chose 42
    loftq_config = peft_loftq_config, # we set None to not use LoFTQ
)

# load dataset
dataset = load_from_disk(dataset_dir)

# include EOS_TOKEN to prevent infinite generation loop
eos_token = tokenizer.eos_token

# format prompts for training and evaluation using the utility function
dataset = dataset.map(
    lambda examples: format_prompts(
        examples,
        input_column=input_column,
        output_column=target,
        eos_token=eos_token,
        for_training=True
    ),
    batched=True
)
dataset["val"] = dataset["val"].map(
    lambda examples: format_prompts(
        examples,
        input_column=input_column,
        for_training=False
    ),
    batched=True,
)

# define metric accumulator for batch evaluation
class MetricAccumulator:
    def __init__(self):
        self.perplexity = 0
        self.rouge_l = 0
        self.bleu = 0
        self.meteor = 0
        self.levenshtein = 0
        self.damerau = 0
        self.jaro_winkler = 0
        self.body_rouge_l = 0
        self.body_bleu = 0
        self.body_meteor = 0
        self.body_levenshtein = 0
        self.body_damerau = 0
        self.body_jaro_winkler = 0
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.valid_json_count = 0
        self.num_samples = 0

    def add(self, loss, inf_model, sample):
        self.num_samples += 1

        # perplexity
        self.perplexity += math.exp(loss.item())

        # inference metrics
        inf_metrics = compute_inference_metrics(sample["text_inf"], max_generation_length, target, sample[target], inf_model, tokenizer)
        self.rouge_l += inf_metrics["Rouge-L"]
        self.bleu += inf_metrics["BLEU"]
        self.meteor += inf_metrics["METEOR"]
        self.levenshtein += inf_metrics["Levenshtein"]
        self.damerau += inf_metrics["Damerau"]
        self.jaro_winkler += inf_metrics["Jaro-Winkler"]

        if target == "json":
            self.body_rouge_l += inf_metrics["body_Rouge-L"]
            self.body_bleu += inf_metrics["body_BLEU"]
            self.body_meteor += inf_metrics["body_METEOR"]
            self.body_levenshtein += inf_metrics["body_Levenshtein"]
            self.body_damerau += inf_metrics["body_Damerau"]
            self.body_jaro_winkler += inf_metrics["body_Jaro-Winkler"]
            self.tp += inf_metrics["TP"]
            self.fp += inf_metrics["FP"]
            self.fn += inf_metrics["FN"]
            self.valid_json_count += inf_metrics["valid_json"]

    def compute(self):

        result = {
            "Perplexity": self.perplexity / self.num_samples,
            "Rouge-L": self.rouge_l / self.num_samples,
            "BLEU": self.bleu / self.num_samples,
            "METEOR": self.meteor / self.num_samples,
            "Levenshtein": self.levenshtein / self.num_samples,
            "Damerau": self.damerau / self.num_samples,
            "Jaro-Winkler": self.jaro_winkler / self.num_samples,
        }

        if target == "json":

            json_scores = {
                "TP": self.tp,
                "FP": self.fp,
                "FN": self.fn,
                "valid_json": self.valid_json_count,
            }

            final_json_scores = compute_final_json_metrics(json_scores, self.num_samples)

            json_results = {
                "body_Rouge-L": self.body_rouge_l / self.valid_json_count,
                "body_BLEU": self.body_bleu / self.valid_json_count,
                "body_METEOR": self.body_meteor / self.valid_json_count,
                "body_Levenshtein": self.body_levenshtein / self.valid_json_count,
                "body_Damerau": self.body_damerau / self.valid_json_count,
                "body_Jaro-Winkler": self.body_jaro_winkler / self.valid_json_count,
                **final_json_scores,
            }

            result.update(json_results)

        return result

# helpers for compute metrics
accumulator = MetricAccumulator()
COUNTER_EVAL = 0

# compute perplexity and inference metrics batch-wise
def compute_metrics(eval_pred, compute_result, current_model):
    global COUNTER_EVAL
    if compute_result:
        COUNTER_EVAL = 0
        return accumulator.compute()
    else:
        accumulator.add(eval_pred.losses, current_model, dataset["val"][COUNTER_EVAL])
        COUNTER_EVAL += 1
        return {}

# resume training from a previous run if requested, otherwise initialize a new run with wandb logging
if run_id:
    wandb.init(project=wandb_project, id=run_id, resume="must")

    # do not evaluate at the beginning if resuming training
    eval_on_start = False

else:
    # create the run name
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{model_name}-{target}{'-cl' if use_cl_loss else ''}-{timestamp}"

    # initialize wandb logging
    run = wandb.init(project=wandb_project, name=run_name, entity=wandb_entity)
    run_id = run.id

    # get previous run ids
    if os.path.exists(wandb_run_ids_file):
        with open(wandb_run_ids_file, "r") as f:
            try:
                run_ids = json.load(f)
            except json.JSONDecodeError:
                run_ids = {}  # handles corrupted or empty file
    else:
        run_ids = {}

    # add the current run id to the file
    run_ids[run_name] = run_id
    with open(wandb_run_ids_file, "w") as f:
        json.dump(run_ids, f, ensure_ascii=False, indent=4)

# set up the run directory based on the run name, and the output base directory
# replaces slashes with dashes to avoid unnecessary subdirectories
run_directory = f"{output_base_dir}/{run_name.replace('/', '-')}"

# set up the logging directory
logging_dir = f"{run_directory}/logs"
os.makedirs(logging_dir, exist_ok=True)

# set up the checkpoint directory
checkpoint_dir = f"{run_directory}/checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# set up the trainer
trainer = CustomTrainer(
    model = model,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
    train_dataset = dataset["train"],
    eval_dataset=dataset["val"],
    dataset_text_field = "text",
    max_seq_length = sequence_length, # we chose 32768 based on our resources
    dataset_num_proc = dataset_num_proc, # we chose 8 based on our resources
    args = TrainingArguments(
        warmup_steps = warmup_steps, # we chose 1000
        num_train_epochs = training_epochs, # we chose 1 epoch since we have many samples and to prevent overfitting
        per_device_train_batch_size = per_device_train_batch_size, # we chose 1 to keep memory as low as possible
        learning_rate = learning_rate, # we set 4e-5 since it is reasonable for fine-tuning
        lr_scheduler_type = lr_scheduler_type, # we chose linear
        fp16 = not is_bfloat16_supported(), # automatically set fp16
        bf16 = is_bfloat16_supported(), # automatically set bf16
        gradient_accumulation_steps = gradient_accumulation_steps, # we chose 1 for no accumulation
        optim = optimizer, # we chose adamw_8bit
        weight_decay = weight_decay, # we chose 0.01
        seed = random_seed, # we chose 42
        eval_strategy = eval_strategy, # we chose steps for logging eval metrics
        eval_steps = eval_steps, # we set 1000
        per_device_eval_batch_size = per_device_eval_batch_size, # we set 1 to keep memory low
        batch_eval_metrics = batch_eval_metrics, # we chose True to save memory
        eval_on_start = eval_on_start, # we chose True for initial metric scores
        include_for_metrics = include_for_metrics, # included loss for easier perplexity calculation
        output_dir = checkpoint_dir,
        save_strategy = save_strategy, # we chose steps
        save_steps = save_steps, # we chose 1000
        logging_dir = logging_dir,
        logging_steps = logging_steps, # we chose 50
        report_to = report_to, # we chose wandb for online logging and tensorboard for local logs
        load_best_model_at_end = load_best_model_at_end, # we set True
        metric_for_best_model = metric_for_best_model, # we chose Rouge-L
        greater_is_better = greater_is_better, # according to Rouge-L, we set True
    ),
)

# training from start or checkpoint
last_checkpoint = get_last_checkpoint(checkpoint_dir)
if last_checkpoint is not None:
    trainer.train(resume_from_checkpoint=last_checkpoint)
else:
    trainer.train()