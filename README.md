# Automatic Ticket Classification

This project focuses on automating the classification of customer complaints for a financial company. The goal is to categorize support tickets based on product and service-related complaints, ensuring quicker response times and improved customer satisfaction.

## Project Overview

The project uses Natural Language Processing (NLP) and machine learning to analyze and classify unstructured complaint text data. By implementing Non-negative Matrix Factorization (NMF) for topic modeling, the model identifies recurring complaint patterns. Supervised learning models are then trained to classify new complaints into predefined categories.

## Dataset

The dataset consists of 78,313 customer complaints in JSON format, each containing various attributes such as complaint text, product type, and additional metadata.

## Project Structure

- `Ravikiran_Automatic_Ticket_Classification.ipynb`: The main notebook containing data loading, cleaning, preprocessing, feature extraction, topic modeling, model training, and evaluation.
- `README.md`: Overview and instructions for the project.

## Key Components

1. **Data Loading and Cleaning**: The data is loaded, and missing values are handled to prepare for analysis.
2. **Text Preprocessing**: Tokenization, lemmatization, and removing stop words enhance text data quality.
3. **Feature Extraction**: TF-IDF vectorization converts text data into numerical features suitable for machine learning.
4. **Topic Modeling**: NMF is applied to cluster complaints into predefined categories:
   - **Credit Card/Prepaid Card**
   - **Bank Account Services**
   - **Theft/Dispute Reporting**
   - **Mortgages/Loans**
   - **Others**
5. **Model Training and Evaluation**: Logistic Regression, Decision Tree, and Random Forest models are trained to classify complaints. The Random Forest model performed the best based on accuracy and evaluation metrics.
6. **Model Inference**: The final model is used to classify new complaint text.

## Insights

- **Data Distribution**: The dataset includes a wide range of complaints across several categories, with "Credit Card/Prepaid Card" and "Bank Account Services" being the most frequent.
- **Text Preprocessing**: Cleaning and lemmatization helped reduce noise, enhancing the quality of text features for topic modeling and classification.
- **Topic Modeling Results**: NMF identified key topics effectively, with words like "credit," "account," "fraud," and "mortgage" prominently defining the topics.
- **Model Performance**: Among the models tested, the Random Forest model achieved the highest classification accuracy, making it suitable for predicting complaint categories.

## Installation and Requirements

To run this project locally, ensure you have the following installed:
- Python 3.6+
- Jupyter Notebook
- Required libraries (listed in `requirements.txt`):
  - pandas
  - numpy
  - scikit-learn
  - spacy
  - nltk

Install the required libraries using:
```bash
pip install -r requirements.txt
