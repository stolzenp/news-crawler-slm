from datasets import load_from_disk
import os
import re
from bs4 import BeautifulSoup, Comment

from common.utils import get_args_from_config

def clean_html(html):
    """Removes <link>, <style>, <script> containing JavaScript, <svg>, <a>, <nav>, <img>, <ins> and <iframe> tags and their content, plus inline styles."""

    soup = BeautifulSoup(html, "html.parser")
    tags_to_remove = ["link", "style", "svg", "a", "nav", "img", "figure", "ins", "iframe", "tickaroo-liveblog", "astro-island"]

    # remove unwanted tags and their content
    for tag in soup(tags_to_remove):
        tag.decompose()

    # remove only JavaScript <script> tags
    for tag in soup.find_all("script"):
        script_type = tag.get("type", "").lower()  # Get type attribute (if exists) and make lowercase
        if not script_type or not "application/ld+json" in script_type:  # delete all unwanted script tags
            tag.decompose()

    # remove divs and sections with ad-related class or id
    for tag in soup.find_all(["div", "section"], class_=lambda c: c and any(
            x in c.lower() for x in ["ad", "advertisement", "sponsored"])):
        tag.decompose()

    for tag in soup.find_all(["div", "section"],
                             id=lambda i: i and any(x in i.lower() for x in ["ad", "sponsored"])):
        tag.decompose()

    # remove inline styles
    for tag in soup.find_all(attrs={"style": True}):
        del tag["style"]

    # process comments individually
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        modified_comment = comment

        # Remove only specified tags inside the comment
        for tag in tags_to_remove:
            modified_comment = re.sub(rf"<{tag}[^>]*>.*?</{tag}>", "", modified_comment,
                                      flags=re.DOTALL)  # remove full tag
            modified_comment = re.sub(rf"<{tag}[^>]*/?>", "", modified_comment,
                                      flags=re.DOTALL)  # remove self-closing tags

        # if the comment is empty after cleaning, remove it completely
        if modified_comment.strip():
            comment.replace_with(modified_comment)  # keep the remaining text
        else:
            comment.extract()

    # remove excessive whitespace and blank lines
    cleaned_html = soup.prettify()

    return cleaned_html.strip()


def clean_dataset(example):
    """Cleans the HTML for a single dataset example."""

    example["html"] = clean_html(example["html"])
    return example

if __name__ == "__main__":
    data_args = get_args_from_config("data_ops_settings")
    split_dataset_path = data_args["split_dataset_directory"]
    clean_dataset_dir = data_args["clean_dataset_directory"]

    dataset = load_from_disk(split_dataset_path)

    # apply cleaning to each dataset split
    for split in ["train", "val", "test"]:
        print(f"Processing {split} split...")
        dataset[split] = dataset[split].map(clean_dataset, desc=f"Cleaning {split}")

    # save cleaned dataset
    os.makedirs(clean_dataset_dir, exist_ok=True)
    dataset.save_to_disk(clean_dataset_dir)
