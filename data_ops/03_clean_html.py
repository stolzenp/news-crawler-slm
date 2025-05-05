from datasets import load_from_disk
import os
import re
from bs4 import BeautifulSoup, Comment

split_dataset_cleaned_dir = "/vol/tmp/stolzenp/training/split_dataset_cleaned_plus"

def clean_html(html):
    """Removes <link>, <style>, <script> containing JavaScript, <svg>, <a>, <nav>, <img>, <ins> and <iframe> tags and their content, plus inline styles."""
    soup = BeautifulSoup(html, "html.parser")
    tags_to_remove = ["link", "style", "svg", "a", "nav", "img", "figure", "ins", "iframe", "tickaroo-liveblog", "astro-island"]

    # Remove unwanted tags and their content
    for tag in soup(tags_to_remove):
        tag.decompose()

    # Remove only JavaScript <script> tags
    for tag in soup.find_all("script"):
        script_type = tag.get("type", "").lower()  # Get type attribute (if exists) and make lowercase
        if not script_type or not "application/ld+json" in script_type:  # delete all unwanted script tags
            tag.decompose()

    # Remove divs and sections with ad-related class or id
    for tag in soup.find_all(["div", "section"], class_=lambda c: c and any(
            x in c.lower() for x in ["ad", "advertisement", "sponsored"])):
        tag.decompose()

    for tag in soup.find_all(["div", "section"],
                             id=lambda i: i and any(x in i.lower() for x in ["ad", "sponsored"])):
        tag.decompose()

    # Remove inline styles
    for tag in soup.find_all(attrs={"style": True}):
        del tag["style"]

    # Process comments individually
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        modified_comment = comment

        # Remove only specified tags inside the comment
        for tag in tags_to_remove:
            modified_comment = re.sub(rf"<{tag}[^>]*>.*?</{tag}>", "", modified_comment,
                                      flags=re.DOTALL)  # Remove full tag
            modified_comment = re.sub(rf"<{tag}[^>]*/?>", "", modified_comment,
                                      flags=re.DOTALL)  # Remove self-closing tags

        # If after cleaning, the comment is empty, remove it completely
        if modified_comment.strip():
            comment.replace_with(modified_comment)  # Keep the remaining text
        else:
            comment.extract()  # Remove the comment if nothing remains

    # Remove excessive whitespace and blank lines
    cleaned_html = soup.prettify()

    return cleaned_html.strip()

dataset = load_from_disk("/vol/tmp/stolzenp/training/split_dataset")

#sample = dataset["train"][35027]

#raw_html = sample["html"]
#cleaned_html = clean_html(raw_html)

#print(cleaned_html)

# Apply cleaning function to each dataset split
def clean_dataset(example):
    example["html"] = clean_html(example["html"])
    return example

for split in ["train", "val", "test"]:
    print(f"Processing {split} split...")
    dataset[split] = dataset[split].map(clean_dataset, desc=f"Cleaning {split}")

# Save cleaned dataset
os.makedirs(split_dataset_cleaned_dir, exist_ok=True)
dataset.save_to_disk(split_dataset_cleaned_dir)
