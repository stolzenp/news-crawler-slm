import os
from typing import Literal
from transformers import HfArgumentParser
from data_structures import Config

# define literal for possible argument categories
ArgumentCategory = Literal["data_extraction_settings", "model_training_settings"]

# set path to config file
path_to_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

# function for accessing arguments of specific category
def get_args_from_config(args: ArgumentCategory):
    """Returns arguments of provided category from config file."""
    parser = HfArgumentParser(Config)
    config = parser.parse_json_file(path_to_config)[0]

    # dynamically access different arguments categories in config
    if hasattr(config, args):
        return getattr(config, args)
    else:
        raise ValueError(f"Unknown argument category: {args}")
