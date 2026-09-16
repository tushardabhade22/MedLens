import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# Load dataset
df = pd.read_csv("../data/reports.csv")

X = df["text"]
y = df["label"]


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# TF-IDF
tfidf = TfidfVectorizer()

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)


# Train Logistic Regression
model = LogisticRegression(max_iter=1000)

model.fit(X_train_tfidf, y_train)


# Test model
y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Save model and TF-IDF vectorizer
joblib.dump(model, "../models/report_classifier.pkl")
joblib.dump(tfidf, "../models/tfidf_vectorizer.pkl")

print("\nModel saved successfully!")
print("TF-IDF vectorizer saved successfully!")