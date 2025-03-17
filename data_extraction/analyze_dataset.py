import json

dataset = []

with open("/vol/tmp/stolzenp/final_dataset.txt", "r", encoding="utf-8") as f:
    for line in f:
        try:
            # Load each line as a JSON object
            entry = json.loads(line.strip())
            dataset.append(entry)
        except json.JSONDecodeError:
            print("Skipping malformed line:", line)

publishers = []
for entry in dataset:
    publisher = entry.get("publisher")
    if publisher in publishers:
        continue
    else:
        publishers.append(publisher)

publishers.sort()

print("Publishers:", publishers)
