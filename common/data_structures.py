from dataclasses import dataclass, field

# dataclasses for argument parsing
@dataclass
class CrawlerArguments:
    publishers_directory: str = field(
        metadata={
            "help": "The directory where the crawler will store the crawled data."
        }
    )
    start_publisher_number: int = field(
        metadata={
            "help": "The publisher number to start crawling from."
        }
    )
    max_articles: int = field(
        metadata={
            "help": "The maximum number of articles to crawl per publisher."
        }
    )
    timeout: int = field(
        metadata={
            "help": "The number of seconds to wait for the next article to be crawled."
        }
    )
    use_cc_news_crawler: bool = field(
        metadata={
            "help": "If set true the CC News Crawler will be used."
        }
    )
    dataset_directory: str = field(
        metadata={
            "help": "The directory where the dataset will be stored."
        }
    )
    dataset_name: str = field(
        metadata={
            "help": "The name of the dataset."
        }
    )
    only_complete_publishers: bool = field(
        metadata={
            "help": (
                "If set to true, only publishers from which the maximum number of articles "
                "was crawled will be considered during dataset creation."
            )
        }
    )

@dataclass
class DataOpsArguments:
    max_input_tokens: int = field(
        metadata={
            "help": "The threshold for the maximum number of input tokens."
        }
    )

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models."
        }
    )
    dataset_path: str = field(
        metadata={
            "help": "Path to the dataset to load."
        }
    )
    output_dir: str = field(
        metadata={
            "help": "The output directory where the model predictions and checkpoints will be written."
        }
    )

@dataclass
class EvaluationArguments:
    target_column: str = field(
        metadata={
            "help": "The target column to evaluate on."
        }
    )

@dataclass
class StatisticsArguments:
    output_dir: str = field(
        metadata={
            "help": "The output directory where the statistics will be written."
        }
    )

@dataclass
class Config:
    data_extraction_settings: CrawlerArguments
    data_ops_settings: DataOpsArguments
    model_training_settings: ModelArguments
    evaluation_settings: EvaluationArguments
    statistics_settings: StatisticsArguments
