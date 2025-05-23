import os

from fundus import PublisherCollection
from tqdm import tqdm

from common.utils import get_args_from_config


def main():
    # get crawler arguments from the config file
    crawler_args = get_args_from_config("data_extraction_settings")

    # get needed arguments
    publishers_directory = crawler_args["publishers_directory"]
    wanted_article_count = crawler_args["max_articles"]
    dataset_directory = crawler_args["dataset_directory"]
    dataset_name = crawler_args["dataset_name"]
    only_complete_publishers = crawler_args["only_complete_publishers"]

    # create the dataset directory if it does not exist
    os.makedirs(dataset_directory, exist_ok=True)

    # raise an error if the publishers directory does not exist
    if not os.path.exists(publishers_directory):
        raise ValueError(
            "The publishers directory does not exist. Please run 'crawl_articles.py' "
            "first to create the publishers directory or set the 'publishers_directory' "
            "setting in the config file to the correct path."
        )

    # raise an error if the wanted article count is not positive
    if wanted_article_count <= 0:
        raise ValueError("The 'max_articles' setting must be positive.")

    # set the path for the final dataset
    output_file = f"{dataset_directory}/{dataset_name}.txt"

    # set variables
    any_matches = False
    publishers = PublisherCollection

    # create a new file for merged data
    with open(output_file, "w") as merged_file:
        # iterate though all possible publisher files
        for publisher in tqdm(publishers, total=len(publishers), desc="Creating Dataset", miniters=1):
            # get publisher filename
            filename = f"{publisher.name.lower()}.txt"

            # check if the file is in the work directory
            if filename in os.listdir(publishers_directory):
                # get the file path
                file_path = os.path.join(publishers_directory, filename)

                # open file
                with open(file_path, "r") as f:
                    # check if only completed publishers should be considered
                    if only_complete_publishers:
                        # count lines without reading the whole file into memory
                        line_count = sum(1 for _ in f)

                        # check if the line count matches the wanted number of articles
                        if line_count == wanted_article_count:
                            # remember matching a file for the final message
                            if not any_matches:
                                any_matches = True

                            # reopen the file to reset the read pointer and write content into a merged file
                            f.seek(0)
                            merged_file.write(f.read())
                    else:
                        # remember matching a file for the final message
                        if not any_matches:
                            any_matches = True

                        # write all available lines into the merged file
                        merged_file.write(f.read())

    # determine the final message based on the merge outcome and requirements
    if only_complete_publishers:
        if any_matches:
            print(f"All data from all fully crawled publishers was merged into '{output_file}'.")
        else:
            print(f"No publisher files found with {wanted_article_count} articles.")
    else:
        if any_matches:
            print(f"All data from all crawled publishers was merged into '{output_file}'.")
        else:
            print(f"No publisher files found in '{publishers_directory}'.")


if __name__ == "__main__":
    main()
