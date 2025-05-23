import re

from common.utils import get_args_from_config


def get_token_counts_for_column(file_path, column_name):
    """Extracts token counts for a specific column from the statistics file."""
    with open(file_path, "r") as g:
        content = g.read()

    # declare the list to collect all token counts for the current column
    column_token_counts = []

    # look for the section related to the specified column
    column_section = re.search(f"Statistics for '{column_name}':(.*?)(?=Statistics for|$)", content, re.DOTALL)

    if column_section:
        # extract the token distribution section for the column
        distribution_section = re.search(r"Token distribution: ({.*?})", column_section.group(1), re.DOTALL)

        if distribution_section:
            # parse the dictionary string and get the token counts
            dist_dict = eval(distribution_section.group(1))

            # repeat the token count (key) as many times as indicated by its value
            for key, value in dist_dict.items():
                column_token_counts.extend([key] * value)

    return column_token_counts


if __name__ == "__main__":
    # get args from the config file
    stats_args = get_args_from_config("statistics_settings")
    output_dir = stats_args["output_dir"]
    limit = stats_args["max_token_limit"]
    stats_file = stats_args["token_stats_file"]
    columns = stats_args["columns"]
    analysis_file = stats_args["distribution_analysis_file"]

    # set paths
    stats_file = f"{output_dir}/{stats_file}"
    analysis_file = f"{output_dir}/{analysis_file}"

    with open(analysis_file, "w") as f:
        f.write(f"Number of samples with more than {limit} tokens: \n\n")

        for column in columns:
            # parse the statistics file for the current column
            token_counts = get_token_counts_for_column(stats_file, column)

            # count how many entries in the current column exceed the threshold
            exceeds_limit_count = sum(1 for count in token_counts if count > limit)

            # track the number of token counts exceeding the max token threshold
            f.write(f"{column}: {exceeds_limit_count}\n")

            print(f"Number of samples in {column} column with more than {limit} tokens: {exceeds_limit_count}")

    print(f"Number of samples with more than {limit} tokens saved to {analysis_file}")
