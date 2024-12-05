# **news-crawler-slm**

**Status**: Work in Progress  
**Project Type**: Master's Thesis Project

## **Overview**
`news-crawler-slm` is a framework designed to assist Small Language Models (SLMs) in extracting relevant information from the HTML content of news articles published by any source. This project leverages the [Fundus framework](https://github.com/flairNLP/fundus), it eases the creation of structured datasets for news articles, which is essential for training and evaluating language models in this domain.

The primary goal of this project is to improve the adaptability of SLMs to various sources of web news content, enabling them to handle diverse styles and structures found across different publishers. This will be achieved by:
- **Finetuning SLMs**: Training models on datasets derived from Fundus articles to improve their extraction accuracy on unfamiliar publishers.
- **Evaluation**: Comparing the performance of these finetuned models against similar language models using a benchmark dataset created with Fundus's manual extraction rules.

## **Features**
- **Data Extraction**: Scripts to extract and preprocess relevant HTML data from Fundus-sourced articles.
- **SLM Finetuning**: Scripts for model training and finetuning on the prepared datasets.
- **Evaluation and Analysis**: Methods to evaluate and compare model performance on unseen publishers.

## **Installation**
1. Clone the repository:
   ```bash
   git clone https://github.com/stolzenp/news-crawler-slm.git
   ```
2. Install dependencies [WIP]:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up the configuration by following the [Configuration](#configuration) section.

## **Usage**  

To perform dataset generation, model training, and evaluation, run the provided scripts. Detailed commands and options are documented in the usage guide.  

### Example Commands  

#### **Dataset Generation**  
Use the following command to generate a dataset:  
```bash
python crawl_articles.py --num_articles=<number_of_articles_per_fundus_publisher> --path=<saving_path_for_dataset>
```  

#### **Model Fine-tuning [WIP]**  
To fine-tune a Small Language Model (SLM), execute:  
```bash
python finetune_model.py --model=<small_language_model_name>
```

#### **Model Evaluation [WIP]**  
To evaluate a resulting model, execute:  
```bash
python evaluate_model.py
```  

## **Configuration [WIP]**
To customize this project:
1. Copy the `.env.example` file and rename it to `.env`.
2. Configure any necessary parameters:
   ```env
   MODEL_CHECKPOINT_PATH=<path_to_model_checkpoint>
   ```

## **Contributing**
This project will welcome contributions after the associated thesis is completed. If you are interested in contributing, please submit an issue or a pull request.

## **License**
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## **Acknowledgments**
- [Fundus Framework](https://github.com/flairNLP/fundus): For providing tools for generating structured datasets essential for model training and evaluation.


