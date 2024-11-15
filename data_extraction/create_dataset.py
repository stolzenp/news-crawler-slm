from fundus import PublisherCollection, Crawler
import json
from tqdm import tqdm


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

# set variable
publishers = PublisherCollection
i = 0
max_articles = 1000

# crawl 1000 articles per publisher and print
with open("/vol/tmp/stolzenp/another_dataset.txt", "w", encoding="utf-8") as f:
    for publisher in publishers:
        if publisher.deprecated:
            continue
        i = i+1
        print(f"{i}: {publisher}")
        crawler = Crawler(publisher)
        for article in tqdm(crawler.crawl(max_articles=max_articles, timeout=120), total=max_articles, desc="Retrieving Articles", miniters=1):
            # collect data
            entry = create_entry(
                article.html.content,
                article.plaintext,
                article.to_json(),
                article.html.requested_url,
                article.publisher,
                article.lang)

            # add the entry as a JSON string to the file
            f.write(json.dumps(entry) + "\n")  # Add newline for separation
