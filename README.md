# IMDB Movie Review Sentiment Analysis

A machine learning project that classifies IMDB movie reviews as **positive** or **negative**.

## Dataset

- File: `IMDB Dataset.csv`
- Columns:
  - `review`: Movie review text
  - `sentiment`: Positive or negative label

> The dataset is excluded from Git because it is large.

## Workflow

1. Load the IMDB review dataset.
2. Clean review text:
   - Convert to lowercase
   - Remove HTML tags and non-letter characters
   - Remove English stop words
   - Apply Porter stemming
3. Split the data into training (80%) and test (20%) sets.
4. Convert text to TF-IDF features with unigrams and bigrams.
5. Train:
   - Multinomial Naive Bayes
   - Logistic Regression
6. Evaluate models using classification reports, ROC-AUC scores, and ROC curves.
7. Display words most associated with positive and negative reviews.

## Data Leakage Prevention

The dataset is split before TF-IDF fitting. The vectorizer learns vocabulary and word-frequency information only from training data:

```python
x_train_tfidf = tfidf.fit_transform(x_train)
x_test_tfidf = tfidf.transform(x_test)
```

This keeps the test set unseen during training and gives a more reliable evaluation.

## Requirements

```bash
pip install pandas scikit-learn matplotlib nltk
```

## Run

Open and run the cells in:

```text
movie_review.ipynb
```
