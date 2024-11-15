import json
from datasets import Dataset

# reading the .txt dataset file and parse the JSON entries
data = []
with open('/vol/tmp/stolzenp/dataset.txt', 'r',  encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))

# first converting the list of dictionaries to a dictionary of lists
columns = {}
for entry in data:
    for key, value in entry.items():
        if key not in columns:
            columns[key] = []
        columns[key].append(value)

# converting the list of dictionaries into a Hugging Face Dataset
dataset = Dataset.from_dict(columns)

print("done")

dataset.push_to_hub(f"fundus-dataset-{len(dataset)}")