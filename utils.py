import os
from typing import Literal, Dict, List, Optional
from transformers import HfArgumentParser
from data_structures import Config

# define literal for possible argument categories
ArgumentCategory = Literal["data_extraction_settings", "model_training_settings", "evaluation_settings"]

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

# Common prompt templates
INPUT_OUTPUT_PROMPT = "Input:\n{}\n\nOutput:\n{}"
INPUT_ONLY_PROMPT = "Input:\n{}\n\nOutput:\n"

def format_prompts(examples: Dict, 
                  input_column: str = "html", 
                  output_column: Optional[str] = None, 
                  eos_token: str = "", 
                  for_training: bool = True,
                  compute_perplexity: bool = True,
                  compute_inference: bool = True) -> Dict:
    """
    Formats prompts for training or evaluation.

    Args:
        examples: Dictionary containing the examples
        input_column: Column name for input data
        output_column: Column name for output data (not used if evaluating inference metrics only)
        eos_token: End of sequence token (needed for training and perplexity evaluation)
        for_training: Whether formatting is for training (True) or evaluation (False)
        compute_perplexity: Whether to compute text_ppl for perplexity evaluation
        compute_inference: Whether to compute text_inf for inference evaluation

    Returns:
        Dictionary with formatted prompts
    """
    inputs = examples[input_column]

    def create_input_output_texts(input_texts, output_texts, end_token):
        """Helper function to create formatted input-output prompts."""
        return [INPUT_OUTPUT_PROMPT.format(i, o) + end_token for i, o in zip(input_texts, output_texts)]

    if for_training:
        # For training, we need both input and output
        if output_column is None:
            raise ValueError("output_column must be provided when for_training=True")

        outputs = examples[output_column]
        return {"text": create_input_output_texts(inputs, outputs, eos_token)}
    else:
        # For evaluation, compute only what's needed
        if output_column is None:
            # For inference metrics only
            return {"text_inf": [INPUT_ONLY_PROMPT.format(i) for i in inputs]}
        else:
            # For evaluation with specified metrics
            result = {}
            outputs = examples[output_column]

            if compute_perplexity:
                result["text_ppl"] = create_input_output_texts(inputs, outputs, eos_token)

            if compute_inference:
                result["text_inf"] = [INPUT_ONLY_PROMPT.format(i) for i in inputs]

            return result
