
# Data Extraction

---

## 📂 Files

| File                | Description                                                      |
|---------------------|------------------------------------------------------------------|
| `crawl_articles.py` | Script for crawling articles by iterating over Fundus publishers |
| `create_dataset.py` | Script for generating dataset based on collected articles        |
| `upload_dataset.py` | Script for uploading dataset to HuggingFace Hub                  |


---

## ▶️ Commands

### `crawl_articles.py`

```bash
python -m data_extraction.crawl_articles
```

### `create_dataset.py`

```bash
python -m data_extraction.create_dataset
```


### `upload_dataset.py`

```bash
python -m data_extraction.upload_dataset
```