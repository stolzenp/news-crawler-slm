import json

dataset = []

with open("/vol/tmp/stolzenp/dataset.txt", "r", encoding="utf-8") as f:
    for line in f:
        try:
            # Load each line as a JSON object
            entry = json.loads(line.strip())
            dataset.append(entry)
        except json.JSONDecodeError:
            print("Skipping malformed line:", line)

language_counts = {}
for entry in dataset:
    language = entry.get("language")
    if language:
        language_counts[language] = language_counts.get(language, 0) + 1

print("Language counts:", language_counts)
