# Spam Email Classifier

This project builds a spam detection model using the Apache SpamAssassin public dataset and compares multiple preprocessing and vectorization choices to improve spam detection performance.

## Project Objective

Build a spam classifier that can distinguish between:

- **Ham** = legitimate email
- **Spam** = unwanted email

The goal was not just to build a working model, but also to improve **recall** while keeping **precision** high.

---

## Dataset

Apache SpamAssassin Public Corpus

- Ham emails: **2551**
- Spam emails: **501**

This is an **imbalanced dataset**, so accuracy alone is not enough.  
Precision and recall are more important.

---

## Steps Implemented

1. Downloaded and extracted SpamAssassin datasets
2. Loaded ham and spam emails into Python
3. Created labels
   - Ham = 0
   - Spam = 1
4. Split data into training and test sets using `train_test_split(..., stratify=y)`
5. Built a custom preprocessing pipeline with optional hyperparameters:
   - Strip email headers
   - Convert to lowercase
   - Replace URLs with `URL`
   - Replace numbers with `NUMBER`
   - Remove punctuation
   - Apply stemming using `PorterStemmer`
6. Converted emails into sparse feature vectors
7. Trained and evaluated multiple model/vectorization combinations
8. Compared results and selected the best-performing approach

---

## Preprocessing Pipeline

The email preprocessing pipeline supports the following options:

- `strip_headers=True`
- `lower_case=True`
- `replace_urls=True`
- `replace_numbers=True`
- `remove_punctuation=True`
- `use_stemming=True`

This helped reduce noise and standardize the text before vectorization.

---

## Models / Experiments

### 1. Baseline Model
**Vectorizer:** `CountVectorizer()`  
**Classifier:** `MultinomialNB()`

**Results:**
- Accuracy: **88.87%**
- Precision: **71.05%**
- Recall: **54.00%**

**Confusion Matrix:**
```text
[[489  22]
 [ 46  54]]


 2. TF-IDF Experiment

Vectorizer: TfidfVectorizer()
Classifier: MultinomialNB()

Results:

Accuracy: 89.03%

Precision: 100.00%

Recall: 33.00%

Confusion Matrix:

[[511   0]
 [ 67  33]]

Observation:
Precision became perfect, but recall dropped significantly.
The model became too conservative and missed too many spam emails.


3. Improved Model with N-grams

Vectorizer: CountVectorizer(ngram_range=(1,2))
Classifier: MultinomialNB()

Results:

Accuracy: 95.74%

Precision: 97.44%

Recall: 76.00%

Confusion Matrix:

[[509   2]
 [ 24  76]]

Observation:
This gave the best balance between precision and recall.
Using bigrams helped the model capture spam phrases such as:

free money

click here

limited offer

buy now

Improvement Summary

| Stage            | Vectorizer / Change       | Accuracy | Precision | Recall |
| ---------------- | ------------------------- | -------- | --------- | ------ |
| Baseline         | CountVectorizer           | 88.87%   | 71.05%    | 54.00% |
| Experiment 2     | TF-IDF                    | 89.03%   | 100.00%   | 33.00% |
| Final Best Model | CountVectorizer + bigrams | 95.74%   | 97.44%    | 76.00% |


Because spam is often identified not only by single words, but by phrases, such as:

free money

limited offer

click now

This improved recall from:

54% → 76%

while also keeping precision very high.
