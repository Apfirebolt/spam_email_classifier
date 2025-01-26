# Spam Email Classifier

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-0.24.2-orange)
![pandas](https://img.shields.io/badge/pandas-1.2.4-green)
![PyQt5](https://img.shields.io/badge/PyQt5-5.15.4-red)

This application uses machine learning techniques to classify emails as spam or not spam. It leverages the Scikit-learn library for building the classification model, pandas for data manipulation, and PyQt5 for the graphical user interface.

## Screenshots

The basic UI showing Email form.

![Screenshot 1](screenshots/1.png)

The next screenshots show emails which are classified as spam and not spam.

![Screenshot 2](screenshots/2.png)
![Screenshot 3](screenshots/3.png)

## Features

- Train a spam classifier using a dataset of labeled emails
- Evaluate the performance of the classifier
- Classify new emails as spam or not spam
- User-friendly GUI for easy interaction using PyQt5

## Algorithms used

```
from sklearn.feature_extraction.text import CountVectorizer
```

`CountVectorizer` is a class from the `sklearn.feature_extraction.text` module in the scikit-learn library. It is used to convert a collection of text documents to a matrix of token counts. This is a crucial step in text preprocessing for machine learning models, as it transforms text data into numerical data that can be used by algorithms. Each row in the resulting matrix represents a document, and each column represents a unique word (token) from the entire corpus. The value in each cell indicates the count of the word in the corresponding document.

```
from sklearn.naive_bayes import MultinomialNB
```

### MultinomialNB

Multinomial Naive Bayes (MultinomialNB) is a variant of the Naive Bayes algorithm that is particularly suited for classification with discrete features (e.g., word counts for text classification). It assumes that the features follow a multinomial distribution. This classifier is often used for document classification problems, where the frequency of each word is used as a feature for training.

Key characteristics:
- Suitable for discrete data.
- Commonly used in text classification and natural language processing (NLP).
- Assumes that the features are conditionally independent given the class.

Parameters:
- `alpha`: Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).

Methods:
- `fit(X, y)`: Fit the model according to the given training data.
- `predict(X)`: Perform classification on an array of test vectors X.
- `predict_proba(X)`: Return probability estimates for the test vector X.
- `score(X, y)`: Returns the mean accuracy on the given test data and labels.

## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## Usage

To start the application, run:

```bash
python app.py
```

## License

This project is licensed under the MIT License.