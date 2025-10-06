from transformers import AutoModelForCausalLM, AutoTokenizer

from common.utils import get_args_from_config

# get dataset arguments from the config file
eval_args = get_args_from_config("evaluation_settings")

# set model variables
model_path = eval_args["model_name_or_path"]
hf_model_path = eval_args["huggingface_model_path"]

# load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# push model and tokenizer to Hugging Face Hub
model.push_to_hub(hf_model_path)
tokenizer.push_to_hub(hf_model_path)

print(f"Model and Tokenizer uploaded to https://huggingface.co/{hf_model_path}")
