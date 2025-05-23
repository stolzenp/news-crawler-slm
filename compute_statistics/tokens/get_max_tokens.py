import re

from common.utils import get_args_from_config


def get_max_tokens_for_column(file_path, column_name):
    """Extracts max token count and sample for a specific column from the statistics file."""

    max_token_count = -1
    max_token_entries = []
    in_target_section = False

    with open(file_path, "r") as g:
        for line in g:
            # detect start of current column section
            if f"Per-sample token counts for '{column_name}':" in line:
                in_target_section = True
                continue

            # break loop if not in the target section anymore
            if in_target_section and "Token Count:" not in line:
                break

            # iterate over samples
            if in_target_section:
                token_match = re.match(r"\s*Token Count: (\d+), Split: (\w+), Position: (\d+)", line)
                if token_match:
                    token_count = int(token_match.group(1))
                    split = token_match.group(2)
                    position = int(token_match.group(3))

                    if token_count > max_token_count:
                        # reset the list if a new max is found
                        max_token_count = token_count
                        max_token_entries = [{"token_count": token_count, "split": split, "position": position}]
                    elif token_count == max_token_count:
                        # add to existing list if same token count
                        max_token_entries.append({"token_count": token_count, "split": split, "position": position})

    return max_token_entries


if __name__ == "__main__":
    # get args from the config file
    stats_args = get_args_from_config("statistics_settings")
    output_dir = stats_args["output_dir"]
    stats_file = stats_args["token_stats_file"]
    columns = stats_args["columns"]
    max_tokens_file = stats_args["max_tokens_file"]

    # set paths
    stats_file = f"{output_dir}/{stats_file}"
    max_tokens_file = f"{output_dir}/{max_tokens_file}"

    with open(max_tokens_file, "w") as f:
        f.write("Maximum token length samples: \n\n")

        for column in columns:
            # get max tokens for the current column
            max_token_samples = get_max_tokens_for_column(stats_file, column)

            # print column info and shared max token count once
            header = (
                f"Column '{column}':\n"
                f"Max Token Count: {max_token_samples[0]['token_count']}\n"
                f"Matching Samples:\n"
            )
            f.write(header)
            print(header, end="")

            # print sample split and index for each max token sample
            for sample in max_token_samples:
                sample_line = f"  - Split: {sample['split']}, Index: {sample['position']}\n"
                f.write(sample_line)
                print(sample_line, end="")

            f.write("\n")
            print()

    print(f"Maximum token samples saved to {max_tokens_file}")
