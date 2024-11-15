import json

# open the text file and read the lines
with open('/vol/tmp/stolzenp/dataset.txt', 'r') as txt_file, open('/vol/tmp/stolzenp/dataset.json', 'w') as json_file:
    for line in txt_file:
        # parse each line and write it directly to the output file
        json_object = json.loads(line.strip())
        json.dump(json_object, json_file)
        json_file.write('\n')  # add newline after each JSON object