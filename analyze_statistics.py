import re

# Threshold for token count
threshold = 16438

# File to read statistics from
stats_file = "token_statistics.txt"


# Function to extract token counts for a specific column (e.g., 'html') from the file
def parse_statistics_for_column(file_path, column_name):
    with open(file_path, "r") as f:
        content = f.read()

    # Find the token counts specifically for the 'html' column
    token_counts = []
    # Look for the section related to the specified column
    column_section = re.search(f"Statistics for '{column_name}':(.*?)(?=Statistics for|$)", content, re.DOTALL)

    if column_section:
        # Extract the token distribution section for the column
        distribution_section = re.search(r"Token distribution: ({.*?})", column_section.group(1), re.DOTALL)

        if distribution_section:
            # Parse the dictionary string and get the token counts
            dist_dict = eval(distribution_section.group(1))
            # Repeat the key as many times as its value
            for key, value in dist_dict.items():
                token_counts.extend([key] * value)

    return token_counts


# Parse the statistics file for the 'html' column
token_counts_html = parse_statistics_for_column(stats_file, 'plain_text')

# Count how many entries in the 'html' column exceed the threshold
exceeds_threshold_html = sum(1 for count in token_counts_html if count > threshold)

# Print result
print(f"Number of samples in 'plain_text' column with more than {threshold} tokens: {exceeds_threshold_html}")
