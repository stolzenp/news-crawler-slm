# **news-crawler-slm**

**Status**: 🚧 Work in Progress  
**Project Type**: Master's Thesis

## 📌 Overview

`news-crawler-slm` is a framework designed to help **Small Language Models (SLMs)** extract relevant information from the HTML content of news articles published by **any** source. Built on top of the [Fundus framework](https://github.com/flairNLP/fundus), it simplifies the creation of structured datasets for training and evaluating language models in diverse news domains.

### 🎯 Objective

Enable SLMs to generalize across varying web news layouts by:
- **Fine-tuning SLMs** on datasets generated from Fundus articles to improve their ability to handle unseen publishers.
- **Evaluating performance** on a benchmark dataset created using Fundus's rule-based extraction, comparing against models that are not fine-tuned.

## ✨ Features

- 🔍 **Data Extraction**: Crawl and extract structured HTML content using Fundus.
- 🧹 **Dataset Operations**: Check and preprocess datasets (e.g., clean messy HTML).
- 📊 **Dataset Statistics**: Compute dataset statistics like token length distribution.
- 🔧 **SLM Fine-tuning**: Fine-tune models on curated datasets.
- 🧪 **Evaluation**: Assess model generalization to unseen publishers.

## 🚀 Installation

**Python version**: `3.10.14` (other versions are not guaranteed to work)

Clone the repository and install dependencies:

```bash
git clone https://github.com/stolzenp/news-crawler-slm.git
cd news-crawler-slm
make install
```

For development:
```bash
make install-dev
```

Then follow the [Configuration](#configuration) instructions.

## ⚙️ Usage

The project is modular, with each feature in a dedicated submodule. Check each module's `README.md` for details. Below are some common commands.

> **Note**: Scripts with `step_x_` prefixes are part of a recommended multi-step preprocessing pipeline.

### 🔄 Preprocessing Pipeline

Preprocessing is broken into multiple steps due to the noisy and lengthy HTML from crawled articles. Each step is executed independently and in order. For example:
- `step_03_clean_html.py`: Cleans HTML content.
- `step_04_get_token_statistics.py`: Computes token stats.
- `step_05_filter_dataset.py`: Filters based on stats.

### 🧪 Example Commands

#### Gather Articles For Dataset
```bash
python -m data_extraction.crawl_articles
```

#### Clean HTML
```bash
python -m data_ops.step_03_clean_html
```

#### Compute Token Statistics
```bash
python -m compute_statistics.tokens.step_04_get_token_statistics
```

#### Fine-tune Model
```bash
python -m model_training.finetune_model
```

#### Evaluate Model
```bash
python -m evaluation.evaluate_model
```

## 🛠️ Configuration

All settings are managed in the `config.json` file (in the project root).

### Steps:
1. Open `config.json`.
2. Locate the section corresponding to your module of interest.
3. Modify values as needed. Refer to `data_structures.py` in the `common` module for detailed schema.

#### Example
```json
"model_name_or_path": "slm-base-model"
```

## 🤝 Contributing

Contributions will be welcomed after the thesis is submitted. Feel free to open an issue or draft a pull request if you're interested.

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [Fundus](https://github.com/flairNLP/fundus): For enabling structured news dataset generation.
