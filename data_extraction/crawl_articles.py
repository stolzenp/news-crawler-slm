from dataclasses import dataclass, field
import json
from tqdm import tqdm
from transformers import HfArgumentParser
from fundus import PublisherCollection, Crawler, CCNewsCrawler

# set dataclasses for argument parsing
@dataclass
class CrawlerArguments:
    use_cc_news_crawler: bool = field(
        metadata={
            "help": "If set true the CC News Crawler will be used."
        }
    )
    start_publisher_number: int = field(
        metadata={
            "help": "The publisher number to start crawling from."
        }
    )
    max_articles: int = field(
        metadata={
            "help": "The maximum number of articles to crawl per publisher."
        }
    )
    timeout: int = field(
        metadata={
            "help": "The number of seconds to wait for the next article."
        }
    )
    work_directory: str = field(
        metadata={
            "help": "The directory where the crawler will store the crawled data."
        }
    )

@dataclass
class Config:
    data_extraction_settings: CrawlerArguments

# function for creating datapoints
def create_entry(html=None, plain_text=None, json_data=None, url=None, publisher_name=None, language=None):
    """Returns data in JSON format."""
    return {
        "html": html,
        "plain_text": plain_text,
        "json": json_data,
        "url": url,
        "publisher": publisher_name,
        "language": language
    }

# set path to config file
path_to_config = "../config.json"

# get crawler arguments from config file
parser = HfArgumentParser(Config)
config = parser.parse_json_file(path_to_config, allow_extra_keys=True)[0]
crawler_args = config.data_extraction_settings


# set meta variables
work_directory = crawler_args["work_directory"]
finished_publisher_file = f"{work_directory}/finished_publishers.txt"

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

# crawl 1000 articles per publisher and print
for publisher in publishers:

    # reset article counter and boolean for publisher completion
    j = 0
    publisher_already_done = False

    # increase publisher counter
    i = i + 1

    # skip publisher prior to start publisher
    if i < start_publisher_number:
        continue

    # check if current publisher was already completed previously
    with open(finished_publisher_file, "r", encoding="utf-8") as h:
        for line in h:
            if line == f"{publisher.name}\n":
                publisher_already_done = True
                break

    # skip publisher if already completed in previous crawler run
    if publisher_already_done:
        continue

    # create file for current publisher
    with open(f"{work_directory}/{publisher.name.lower()}.txt", "w", encoding="utf-8") as f:
        if publisher.deprecated:
            continue
        print(f"{i}: {publisher}")

        # set crawler type
        if use_cc_news_crawler:
            crawler = CCNewsCrawler(publisher)
        else:
            crawler = Crawler(publisher)

        # crawl articles for current publisher
        for article in tqdm(crawler.crawl(max_articles=max_articles, timeout=2000), total=max_articles, desc="Retrieving Articles", miniters=1):

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