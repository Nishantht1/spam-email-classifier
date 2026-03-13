import re
from pathlib import Path
from nltk.stem import PorterStemmer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

ham_dir=Path("data/easy_ham")
spam_dir=Path("data/spam")

ham_emails=[]
for file in ham_dir.iterdir():
    if file.is_file():
        text=file.read_text(encoding="latin-1")
        ham_emails.append(text)

spam_emails=[]
for file in spam_dir.iterdir():
    if file.is_file():
        text=file.read_text(encoding="latin-1")
        spam_emails.append(text)

print(f"Number of ham emails: {len(ham_emails)}")
print(f"Number of spam emails: {len(spam_emails)}")

print("\nFirst HAM email snippet:\n")
print(ham_emails[0][:500])

print("\nFirst SPAM email snippet:\n")
print(spam_emails[0][:500])



X=ham_emails+ spam_emails
y=[0]*len(ham_emails)+[1]*len(spam_emails)  

print(f"\nTotal emails: {len(X)}")
print(f"Total labels: {len(y)}")

print("first 10 labels:", y[:10])
print("last 10 labels:", y[-10:])

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)


print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")  
print(f"Training labels distribution: {sum(y_train)} spam, {len(y_train)-sum(y_train)} ham")
print(f"Test labels distribution: {sum(y_test)} spam, {len(y_test)-sum(y_test)} ham")

print(y_train[:10]) 
print(y_test[:10])

stemmer = PorterStemmer()

vectorizer= CountVectorizer(ngram_range=(1,2))
# X_train_vectorized=vectorizer.fit_transform(X_train)
# X_test_vectorized=vectorizer.transform(X_test)
# print(X_train_vectorized.shape)
# print(X_test_vectorized.shape)
# print(len(vectorizer.vocabulary_))

def preprocess_email(email, strip_headers=True, lower_case=True,replace_urls=True, replace_numbers=True,remove_punctuation=True,use_stemming=True):
     text = email
     if strip_headers:
        parts = text.split("\n\n", 1)
        if len(parts) == 2:
            text = parts[1]

     if lower_case:
        text = text.lower()
    

     if replace_urls:
        url_pattern = r"(http[s]?://\S+|www\.\S+)"
        text = re.sub(url_pattern, " URL ", text)

     

     if replace_numbers:
        number_pattern = r"\d+(\.\d+)?"
        text = re.sub(number_pattern, " NUMBER ", text)
    
     if remove_punctuation:
        punct_pattern = r"[^\w\s]"
        text = re.sub(punct_pattern, " ", text)

     if use_stemming:
        words = text.split()
        stemmed_words = [stemmer.stem(word) for word in words]
        text = " ".join(stemmed_words)

     return text.strip()

sample = ham_emails[0]  

processed = preprocess_email(sample)

print("ORIGINAL (first 300 chars):")
print(sample[:300])

print("\nPROCESSED (first 300 chars):")
print(processed[:300])



X_train_clean = [preprocess_email(email) for email in X_train]
X_test_clean = [preprocess_email(email) for email in X_test]

X_train_vectorized = vectorizer.fit_transform(X_train_clean)
X_test_vectorized = vectorizer.transform(X_test_clean)


print(X_train_vectorized.shape)
print(X_test_vectorized.shape)
print(len(vectorizer.vocabulary_))

model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

y_pred = model.predict(X_test_vectorized)

print(y_pred[:20])
print(y_test[:20])


print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))