from transformers import AutoTokenizer, AutoModelForCausalLM

from common.utils import get_args_from_config

# get dataset arguments from the config file
model_args = get_args_from_config("data_extraction_settings")

# set model variables
model_path = model_args["model_checkpoint_directory"]
hf_model_path = model_args["hf_model_path"]

# load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("your_model_dir")
tokenizer = AutoTokenizer.from_pretrained("your_model_dir")

# push model and tokenizer to Hugging Face Hub
model.push_to_hub("your-username/your-model-name")
tokenizer.push_to_hub("your-username/your-model-name")

print(f"Model and Tokenizer uploaded to https://huggingface.co/models/{hf_model_path}")