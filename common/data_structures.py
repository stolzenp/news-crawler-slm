from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union


# dataclasses for argument parsing
@dataclass
class CrawlerArguments:
    publishers_directory: str = field(metadata={"help": "The directory where the crawler will store the crawled data."})
    start_publisher_number: int = field(metadata={"help": "The publisher number to start crawling from."})
    max_articles: int = field(metadata={"help": "The maximum number of articles to crawl per publisher."})
    timeout: int = field(metadata={"help": "The number of seconds to wait for the next article to be crawled."})
    use_cc_news_crawler: bool = field(metadata={"help": "If set true the CC News Crawler will be used."})
    dataset_directory: str = field(metadata={"help": "The directory where the dataset will be stored."})
    dataset_name: str = field(metadata={"help": "The name of the dataset."})
    only_complete_publishers: bool = field(
        metadata={
            "help": (
                "If set to true, only publishers from which the maximum number of articles "
                "was crawled will be considered during dataset creation."
            )
        }
    )
    huggingface_dataset_path: Optional[str] = field(
        default=None, metadata={"help": "The path to the HuggingFace dataset to load."}
    )


@dataclass
class DataOpsArguments:
    source_dataset_path: str = field(
        metadata={"help": "Path to the dataset text file to convert to HuggingFace dataset format."}
    )
    target_dataset_directory: str = field(
        metadata={"help": "The directory where the dataset in HuggingFace format will be stored."}
    )
    training_split_size: float = field(metadata={"help": "The proportion of the target dataset to use for training."})
    seed: int = field(metadata={"help": "The seed for randomly shuffling publishers in the dataset."})
    split_dataset_directory: str = field(metadata={"help": "The directory where the split dataset will be stored."})
    clean_dataset_directory: str = field(
        metadata={"help": "The directory where the dataset with cleaned HTML will be stored."}
    )
    token_stats_file: str = field(
        metadata={"help": "Path to the file containing token statistics for filtering out excessively long samples."}
    )
    html_token_threshold: int = field(metadata={"help": "The maximum number of tokens allowed in a sample's HTML."})
    target_token_threshold: int = field(
        metadata={"help": "The maximum number of tokens allowed in a sample's target columns."}
    )
    filtered_dataset_directory: str = field(
        metadata={"help": "The directory for the filtered dataset (based on token statistics and thresholds)."}
    )
    samples_per_publisher: int = field(
        metadata={"help": "The number of samples per publisher to keep in the shrinked dataset."}
    )
    shrinked_dataset_directory: str = field(
        metadata={"help": "The directory where the shrinked dataset (reduced samples per publisher) will be stored."}
    )

    # check sample settings
    dataset_directory: str = field(metadata={"help": "The directory of the dataset to check samples for."})
    base_output_dir: str = field(metadata={"help": "The base output directory for checked samples."})
    columns: List[str] = field(metadata={"help": "The columns to include when checking samples."})
    split: str = field(metadata={"help": "The split of the sample to check."})
    sample_id: int = field(metadata={"help": "The sample ID of the sample to check."})


@dataclass
class ModelArguments:
    split_dataset_directory: str = field(metadata={"help": "Path to the preprocessed and split dataset."})
    model_name_or_path: str = field(metadata={"help": "Pretrained model name or path to local model."})
    output_base_dir: str = field(metadata={"help": "Base output directory to save model checkpoints and logs."})
    use_contrastive_loss: bool = field(metadata={"help": "Whether to apply contrastive loss during training."})
    input_column: str = field(metadata={"help": "Name of the column containing input text (e.g., HTML)."})
    target_column: str = field(metadata={"help": "Name of the column containing target text (e.g., plaintext)."})
    sequence_length: int = field(metadata={"help": "Maximum length of model input sequences."})
    max_generation_length: int = field(metadata={"help": "Maximum number of tokens to generate."})

    # PEFT settings
    peft_r: int = field(metadata={"help": "LoRA rank (r)."})
    peft_target_modules: List[str] = field(metadata={"help": "List of model modules to apply LoRA to."})
    peft_lora_alpha: int = field(metadata={"help": "LoRA alpha scaling factor."})
    peft_lora_dropout: float = field(metadata={"help": "LoRA dropout rate."})
    peft_bias: str = field(metadata={"help": "LoRA bias handling: 'none', 'all', or 'lora_only'."})
    peft_use_gradient_checkpointing: Optional[Union[str, bool]] = field(
        metadata={"help": "Use gradient checkpointing to save memory. Can be 'unsloth', True, or False."}
    )
    peft_loftq_config: Optional[Dict] = field(metadata={"help": "Configuration for LoFT-Q quantization (if used)."})

    # dataset & training setup
    dataset_num_proc: int = field(metadata={"help": "Number of processes for dataset preprocessing."})
    warmup_steps: int = field(metadata={"help": "Number of warmup steps before learning rate decay begins."})
    training_epochs: int = field(metadata={"help": "Total number of training epochs."})
    per_device_train_batch_size: int = field(metadata={"help": "Training batch size per device."})
    learning_rate: float = field(metadata={"help": "Initial learning rate."})
    learning_rate_scheduler_type: str = field(metadata={"help": "Learning rate scheduler type (e.g., linear, cosine)."})
    gradient_accumulation_steps: int = field(
        metadata={"help": "Number of steps to accumulate gradients before backward/update."}
    )
    optimizer: str = field(metadata={"help": "Optimizer to use (e.g., adamw_8bit)."})
    weight_decay: float = field(metadata={"help": "Weight decay for optimizer."})
    seed: int = field(metadata={"help": "Random seed for reproducibility."})
    load_in_4bit: bool = field(metadata={"help": "Load model weights in 4-bit precision for memory efficiency."})

    # evaluation
    eval_strategy: str = field(metadata={"help": "Evaluation strategy ('steps' or 'epoch')."})
    eval_steps: int = field(metadata={"help": "Number of steps between evaluations."})
    per_device_eval_batch_size: int = field(metadata={"help": "Evaluation batch size per device."})
    batch_eval_metrics: bool = field(
        metadata={"help": "Controls if eval metrics calculation is split into batches to save memory."}
    )
    eval_on_start: bool = field(metadata={"help": "Run evaluation before starting training."})
    include_for_metrics: List[str] = field(
        metadata={"help": "List of keys to include for metric computation (e.g., ['loss'])."}
    )

    # checkpointing & logging
    save_strategy: str = field(metadata={"help": "Checkpoint saving strategy ('steps' or 'epoch')."})
    save_steps: int = field(metadata={"help": "Number of steps between checkpoints."})
    logging_steps: int = field(metadata={"help": "Number of steps between logging events."})
    report_to: List[str] = field(metadata={"help": "Platforms to report metrics to (e.g., ['wandb', 'tensorboard'])."})

    # best model tracking
    load_best_model_at_end: bool = field(metadata={"help": "Whether to load the best model found during training."})
    metric_for_best_model: str = field(metadata={"help": "Metric used to determine the best model (e.g., 'Rouge-L')."})
    greater_is_better: bool = field(metadata={"help": "Whether a higher value of the best metric is better."})

    # WandB tracking
    enable_wandb: bool = field(metadata={"help": "Whether to enable Weights & Biases logging."})
    wandb_project: Optional[str] = field(metadata={"help": "Weights & Biases project name."})
    wandb_entity: Optional[str] = field(metadata={"help": "Weights & Biases entity (team or user)."})
    wandb_run_name: Optional[str] = field(
        metadata={"help": "Name of the Weights & Biases run to resume if Weights & Biases is enabled."}
    )


@dataclass
class EvaluationArguments:
    model_name_or_path: str = field(
        metadata={"help": ("Path to the model checkpoint or Hugging Face model to evaluate.")}
    )
    sequence_length: int = field(metadata={"help": ("Maximum sequence length for model inputs during evaluation.")})
    max_generation_length: int = field(metadata={"help": ("Maximum number of tokens to generate during evaluation.")})
    dataset_path: str = field(metadata={"help": ("Path to the dataset used for evaluation.")})
    split: str = field(metadata={"help": ("Which split of the dataset to evaluate on (e.g., 'val' or 'test').")})
    run_all_metrics: bool = field(metadata={"help": ("Whether to compute all available metrics during evaluation.")})
    target_column: str = field(
        metadata={"help": ("Column name in the dataset that contains the target output (e.g., 'json').")}
    )
    base_output_dir: str = field(metadata={"help": ("Directory where evaluation outputs and metrics will be saved.")})
    raw_metrics_file: str = field(metadata={"help": ("Filename to save raw metric results.")})
    huggingface_model_path: Optional[str] = field(
        metadata={"help": ("Optional Hugging Face model hub path (e.g., 'username/model-name').")}
    )
    log_to_wandb: bool = field(metadata={"help": ("Whether to log evaluation metrics to Weights & Biases.")})
    wandb_project: Optional[str] = field(
        default=None, metadata={"help": ("Weights & Biases project name for logging.")}
    )
    wandb_entity: Optional[str] = field(default=None, metadata={"help": ("Weights & Biases entity (team or user).")})
    wandb_run_name: Optional[str] = field(
        default=None, metadata={"help": ("Weights & Biases run name to log this evaluation run to.")}
    )


@dataclass
class StatisticsArguments:
    output_dir: str = field(metadata={"help": ("Directory where all generated statistics will be saved.")})
    dataset_directory: str = field(metadata={"help": ("Path to the directory containing the dataset to analyze.")})
    token_stats_file: str = field(metadata={"help": ("Filename for saving token-level statistics.")})
    dataset_stats_file: str = field(
        metadata={"help": ("Filename for saving statistics about dataset splits (e.g., number of samples per split).")}
    )
    language_stats_file: str = field(
        metadata={"help": ("Filename for saving statistics about language distribution in the dataset.")}
    )
    max_token_limit: int = field(metadata={"help": ("Token threshold used for counting samples exceeding this limit.")})
    columns: List[str] = field(
        metadata={"help": ("List of dataset columns to involve (e.g., ['html', 'plaintext', 'json']).")}
    )
    distribution_analysis_file: str = field(
        metadata={
            "help": (
                "Filename for saving results of distribution analysis, such as count of samples exceeding token limit."
            )
        }
    )
    max_tokens_file: str = field(metadata={"help": ("Filename for saving sample with maximum number of tokens.")})


@dataclass
class Config:
    data_extraction_settings: CrawlerArguments
    data_ops_settings: DataOpsArguments
    model_training_settings: ModelArguments
    evaluation_settings: EvaluationArguments
    statistics_settings: StatisticsArguments
