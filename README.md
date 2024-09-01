# Fake News Detection Project

This project involves detecting fake news articles using machine learning techniques. The dataset is preprocessed and then classified using various models. The project contains two primary Jupyter Notebooks:

1. **`FakeNewsCountVectorizer.ipynb`**: Implements a classifier using Count Vectorization.
2. **`FakeNewsTFIDF.ipynb`**: Implements a classifier using TF-IDF vectorization.

## Files

### 1. `FakeNewsCountVectorizer.ipynb`

This Jupyter Notebook uses Count Vectorization for classifying news articles as fake or real. The main steps include:

- **Loading the dataset**: Reads the CSV file containing news articles and their labels.
- **Preprocessing the data**: Cleans and processes the text data by removing non-alphabetic characters, converting text to lowercase, and applying stemming.
- **Vectorization**: Transforms the text data into numerical features using Count Vectorizer.
- **Model Training and Evaluation**: Trains a Multinomial Naive Bayes classifier and evaluates its performance using accuracy and confusion matrix.
- **Additional Models**: Includes implementation of Passive Aggressive Classifier and hyperparameter tuning for Multinomial Naive Bayes.

#### Dependencies

- `pandas`
- `numpy`
- `scikit-learn`
- `nltk`

### 2. `FakeNewsTFIDF.ipynb`

This Jupyter Notebook uses TF-IDF (Term Frequency-Inverse Document Frequency) vectorization for classifying news articles. The main steps include:

- **Loading the dataset**: Reads the CSV file containing news articles and their labels.
- **Preprocessing the data**: Similar to the Count Vectorization notebook, this includes text cleaning, lowercase conversion, and stopword removal.
- **Vectorization**: Transforms the text data into numerical features using TF-IDF Vectorizer.
- **Model Training and Evaluation**: Trains a chosen classifier (such as Naive Bayes or any other classifier) and evaluates its performance.

#### Dependencies

- `pandas`
- `numpy`
- `scikit-learn`
- `nltk`

