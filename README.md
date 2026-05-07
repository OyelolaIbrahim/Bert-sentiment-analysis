**BERT-Based Sentiment Analysis with PyTorch**

![Python](https://img.shields.io/badge/Python-3.x-blue)  ![Framework](https://img.shields.io/badge/Framework-PyTorch-blue)  ![Model](https://img.shields.io/badge/Model-BERT-blue)  ![Task](https://img.shields.io/badge/Task-Sentiment%20Classification-blue)  
## Overview
Fine-tunes a pre-trained BERT (bert-base-cased) transformer model for binary sentiment classification using PyTorch and the HuggingFace Transformers library. Implements a custom training loop, Stratified K-Fold cross-validation, and evaluates performance using F1-score.
## Dataset

- **Name:** IMDB Movie Reviews Sentiment Dataset
- **Source:** [Download from Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- **Size:** 50,000 movie reviews (25,000 positive, 25,000 negative)
- **Classes:** Positive (1), Negative (0)
- **Format:** CSV file with two columns — `review` (raw text) and `sentiment` (positive/negative)
- **Instructions:** Download `IMDB Dataset.csv` from the link above and place it in the root folder of this project before running the notebook.
## Problem Statement
Classify text as positive or negative sentiment by fine-tuning a pre-trained BERT transformer model. Demonstrates expertise in transfer learning with large language models for downstream NLP classification tasks.
## Approach
-	Tokenised input text using HuggingFace BertTokenizer with padding and truncation to a fixed max length.
-	Built a custom PyTorch Dataset class and DataLoader pipeline for efficient batch processing.
-	Loaded the pre-trained bert-base-cased model and added a classification head for binary output.
-	Trained using Adam optimiser with a learning rate scheduler for gradual learning rate decay.
-	Applied Stratified K-Fold cross-validation to ensure balanced class representation across all folds.
-	Evaluated model performance using F1-score and cross-entropy loss metrics.
## Results
| Property            | Details                         |
| ------------------- | ------------------------------- |
| Base Model          | bert-base-cased (HuggingFace)   |
| Task                | Binary Sentiment Classification |
| Validation Strategy | Stratified K-Fold               |
| Optimizer           | Adam + LR Scheduler             |
| Metric              | F1-Score                        |
| Loss Function       | Cross-Entropy                   |



## Technologies
Python, PyTorch, HuggingFace Transformers, Scikit-Learn, NumPy, Pandas
## How to Run

```bash
git clone https://github.com/OyelolaIbrahim/bert-sentiment-analysis.git
cd bert-sentiment-analysis
pip install torch transformers scikit-learn pandas numpy
jupyter notebook bert_sentiment_analysis.ipynb

