import json
import argparse
import os

# /vol/tmp/stolzenp/results/NewsCrawlerSLM_cl_loss/plaintext/inference_metrics_shrinked_results.json
# /vol/tmp/stolzenp/results/ReaderLM-v2_finetuned_cl_loss/plaintext/inference_metrics_shrinked_results.json

def filter_results(input_path, output_path=None, mean_output=None):

    dir_path = os.path.dirname(input_path)

    if output_path is None:
        output_path = dir_path + "/filtered_inference_metrics_results.json"

    if mean_output is None:
        mean_output = dir_path + "/filtered_inference_metrics_means.txt"

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)  # assuming list of JSON objects

    # all observed Damerau above 10000 were degenerations so we exclude these samples
    filtered_data = [obj for obj in data if obj.get("Damerau") < 10000]

    numeric_attributes = [key for obj in filtered_data for key, value in obj.items() if isinstance(value, (int, float))]

    # calculate means for each numerical attribute
    if filtered_data:

        # save filtered results
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(filtered_data, f, indent=4)

        print(f"Filtered results saved to {output_path}\n")

        means = {
            attr: sum(obj[attr] for obj in filtered_data) / len(filtered_data)
            for attr in numeric_attributes
        }

        # save new mean values
        with open(mean_output, "w", encoding="utf-8") as f:
            print("Mean values for filtered results:\n")
            for metric, mean_value in means.items():
                message = f"{metric} mean: {mean_value}\n"
                f.write(message)
                print(message)

        print(f"New mean values saved to {mean_output}\n")

    else:
        print("No elements remain after filtering.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument( "-i", "--input", type=str, required=True, help="Inference Metrics scores file")
    parser.add_argument( "-o", "--output", type=str, help="Filtered file path (.json)")
    parser.add_argument( "-m", "--mean_output", type=str, help="Mean values path (.txt)")

    args = parser.parse_args()
    filter_results(args.input, args.output, args.mean_output)

