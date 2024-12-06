from dataclasses import dataclass, field

# dataclasses for argument parsing
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
class CrawlerArguments:
    use_cc_news_crawler: bool = field(
        metadata={
            "help": "If set true the CC News Crawler will be used."
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
            "help": "The number of seconds to wait for the next article."
        }
    )
    work_directory: str = field(
        metadata={
            "help": "The directory where the crawler will store the crawled data."
        }
    )
    dataset_name: str = field(
        metadata={
            "help": "The name of the dataset."
        }
    )
    only_complete_publishers: bool = field(
        metadata={
            "help": "If set to true, only publishers from which the maximum number of articles was crawled will be considered during dataset creation."
        }
    )

@dataclass
class Config:
    model_training_settings: ModelArguments
    data_extraction_settings: CrawlerArguments