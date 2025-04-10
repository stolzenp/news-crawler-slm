from unsloth import FastLanguageModel
from transformers import TextStreamer
from datasets import load_from_disk

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "/vol/tmp/stolzenp/training/qwen2.5-1.5b_24k+8k/results/plaintext/checkpoint-70458", # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length = 32768,
    dtype = None,
    load_in_4bit = True,
)

prompt = """
Input:
{}

Output:
"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    inputs       = examples["html"]
    outputs      = examples["plain_text"]
    texts = []
    for input in inputs:
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = prompt.format(input)
        text = tokenizer(text, return_tensors="pt")
        texts.append(text)
    return { "text" : texts, }
pass

dataset = load_from_disk("/vol/tmp/stolzenp/training/split_dataset_cleaned_filtered_24k")
test_set = dataset["test"]
#test_set = test_set.map(formatting_prompts_func, batched = True,)

inputs = tokenizer("""Input:{"""+ test_set[0]["html"] + """}
Output:""", return_tensors="pt")

FastLanguageModel.for_inference(model) # Enable native 2x faster inference
text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 8192)
print(test_set[0]["plain_text"])