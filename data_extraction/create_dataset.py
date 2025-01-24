import os
from tqdm import tqdm
from fundus import PublisherCollection
from utils import get_args_from_config

# get crawler arguments from config file
crawler_args = get_args_from_config("data_extraction_settings")

# get needed arguments
wanted_article_count = crawler_args["max_articles"]
directory = crawler_args["work_directory"]
dataset_name = crawler_args["dataset_name"]
only_complete_publishers = crawler_args["only_complete_publishers"]

# raise error if directory does not exist
if not os.path.exists(directory):
    raise ValueError("The work directory does not exist.")

# set path for the final dataset
output_file = f"{directory}/{dataset_name}.txt"

# set variables
any_matches = False
publishers = PublisherCollection

# create new file for merged data
with open(output_file, 'w') as merged_file:

    # iterate though all possible publisher files
    for publisher in tqdm(publishers, total=len(publishers), desc="Creating Dataset", miniters=1):

        # get publisher filename
        filename = f"{publisher.name.lower()}.txt"

        # check if file is in work directory
        if filename in os.listdir(directory):

            # get file path
            file_path = os.path.join(directory, filename)

            # open file
            with open(file_path, 'r') as f:

                # check if only completed publishers should be considered
                if only_complete_publishers:

                    # count lines without reading the whole file into memory
                    line_count = sum(1 for _ in f)

                    # check if line count matches the wanted amount of articles
                    if line_count == wanted_article_count:

                        # remember matching file
                        if not any_matches:
                            any_matches = True

                        # reopen the file to reset the read pointer and write content into merged file
                        f.seek(0)
                        merged_file.write(f.read())
                else:

                    # remember matching file
                    if not any_matches:
                        any_matches = True

                    # write all available lines into merged file
                    merged_file.write(f.read())

# determine final message based on the merge outcome and requirements
if only_complete_publishers:
    if any_matches:
        print(f"All data from all fully crawled publishers was merged into '{output_file}'.")
    else:
        print(f"No publisher files found with {wanted_article_count} articles.")
else:
    if any_matches:
        print(f"All data from all crawled publishers was merged into '{output_file}'.")
    else:
        print(f"No publisher files found in work directory '{directory}'.")
