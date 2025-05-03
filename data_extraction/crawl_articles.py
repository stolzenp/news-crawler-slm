import json
import os

from tqdm import tqdm
from fundus import PublisherCollection, Crawler, CCNewsCrawler

from utils import get_args_from_config

# function for creating datapoints
def create_entry(html=None, plaintext=None, json_data=None, url=None, publisher_name=None, language=None):
    """Returns data in JSON format."""

    return {
        "html": html,
        "plaintext": plaintext,
        "json": json_data,
        "url": url,
        "publisher": publisher_name,
        "language": language
    }

def main():
    # get crawler arguments from the config file
    crawler_args = get_args_from_config("data_extraction_settings")

    # set meta variables
    publishers_directory = crawler_args["publishers_directory"]
    finished_publisher_file = f"{publishers_directory}/finished_publishers.txt"

    # set crawler variables
    publishers = PublisherCollection
    start_publisher_number = crawler_args["start_publisher_number"]
    max_articles = crawler_args["max_articles"]
    timeout = crawler_args["timeout"]
    use_cc_news_crawler = crawler_args["use_cc_news_crawler"]

    # set loop variables
    i = 0
    j = 0
    publisher_already_done = False

    # crawl max_articles per publisher and print
    for publisher in publishers:
        # increase publisher counter
        i = i + 1

        # skip the current publisher if it is preceding to the start publisher
        if i < start_publisher_number:
            continue

        # skip deprecated publishers
        if publisher.deprecated:
            continue

        # reset boolean for publisher completion
        publisher_already_done = False

        # check if the current publisher was already completed previously
        if os.path.isfile(finished_publisher_file):
            with open(finished_publisher_file, "r", encoding="utf-8") as h:
                for line in h:
                    if line == f"{publisher.name}\n":
                        publisher_already_done = True
                        break

        # skip publisher if already completed in a previous crawler run
        if publisher_already_done:
            continue

        # set the path for the publisher file
        publisher_file = f"{publishers_directory}/{publisher.name.lower()}.txt"

        # create directory if necessary
        os.makedirs(os.path.dirname(publisher_file), exist_ok=True)

        # create a file for the current publisher
        with open(publisher_file, "w", encoding="utf-8") as f:
            # reset article counter
            j = 0

            # log currently crawled publisher
            print(f"{i}: {publisher}")

            # set crawler type
            if use_cc_news_crawler:
                crawler = CCNewsCrawler(publisher)
            else:
                crawler = Crawler(publisher)

            # crawl articles for the current publisher
            for article in tqdm(crawler.crawl(max_articles=max_articles, timeout=timeout), total=max_articles, desc="Retrieving Articles", miniters=1):
                # increase article counter
                j = j + 1

                # collect data
                entry = create_entry(
                    article.html.content,
                    article.plaintext,
                    article.to_json(),
                    article.html.requested_url,
                    article.publisher,
                    article.lang)

                # add the entry as a JSON string to the file
                f.write(json.dumps(entry) + "\n")

                # remember completed publishers
                if j == max_articles:
                    with open(finished_publisher_file, "a", encoding="utf-8") as g:
                        g.write(f"{publisher.name}\n")

if __name__ == "__main__":
    main()