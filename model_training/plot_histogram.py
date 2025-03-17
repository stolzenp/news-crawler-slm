import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Step 1: Extract the first token distribution inside {}
token_counts = []

with open('/vol/fob-vol4/mi17/stolzenp/news-crawler-slm/token_statistics_cleaned_html.txt', 'r') as f:
    content = f.read()

    # Regular expression to match the first occurrence of a token distribution inside { ... }
    pattern = r'\{([^}]+)\}'  # Match everything inside the first set of curly braces
    match = re.search(pattern, content)

    if match:
        # Extract the token distribution string and split it into individual token:count pairs
        distribution_str = match.group(1)

        # Process each token:count pair
        pairs = distribution_str.split(',')  # Split by comma to get individual pairs
        for pair in pairs:
            token, count = pair.split(':')
            token = int(token.strip())
            count = int(count.strip())

            # Add the token to the list as many times as indicated by the count
            token_counts.extend([token] * count)

# Step 2: Calculate mean and median
mean = np.mean(token_counts)
mean = float(mean)
median = np.median(token_counts)
median = float(median)

def format_number(x):
    return f'{x:,.0f}'

plt.figure(figsize=(10, 6))

plt.hist(token_counts, bins=90, edgecolor='black', alpha=0.7, color="skyblue")

plt.axvline(mean, color='red', linestyle='dashed', linewidth=2, label=f"Mean: {format_number(mean)}")
plt.axvline(median, color='green', linestyle='dashed', linewidth=2, label=f"Median: {format_number(median)}")

plt.xticks(rotation=45)

formatter = FuncFormatter(lambda x, _: f'{x:,.0f}')
plt.gca().xaxis.set_major_formatter(formatter)
plt.gca().yaxis.set_major_formatter(formatter)

plt.xlabel("# Tokens")
plt.ylabel("Frequency")
plt.title(f"Token Count Distribution (html)")
plt.legend()
plt.tight_layout()

plt.savefig(f"token_distribution_cleaned_html_html_fancy.png")
plt.close()
