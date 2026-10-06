import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
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


# Create combined TF-IDF + Logistic Regression model
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression(max_iter=1000))
])


# Train model
model.fit(X_train, y_train)


# Test model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Save combined model with compression
joblib.dump(
    model,
    "../models/medlens_model.pkl",
    compress=3
)

print("\nMedLens model saved successfully!")
print("File: models/medlens_model.pkl")