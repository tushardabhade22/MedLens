import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# Load dataset
df = pd.read_csv("../data/reports.csv")

X = df["text"]
y = df["label"]


# Split
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


# Model
model = LogisticRegression(
    max_iter=1000
)

model.fit(
    X_train_tfidf,
    y_train
)


# New unseen reports
while True:

    report = input("\nEnter medical report text (or type 'exit'): ")

    if report.lower() == "exit":
        break

    report_tfidf = tfidf.transform([report])

    prediction = model.predict(report_tfidf)[0]

    probability = model.predict_proba(report_tfidf)[0]
    confidence = probability.max()

    print("\nPredicted Report Type:", prediction)
    print("Confidence:", round(confidence * 100, 2), "%")


# Convert new reports using trained TF-IDF
new_reports_tfidf = tfidf.transform(new_reports)


# Predict
predictions = model.predict(new_reports_tfidf)

# Confidence
probabilities = model.predict_proba(new_reports_tfidf)


for report, prediction, probability in zip(
    new_reports,
    predictions,
    probabilities
):

    confidence = probability.max()

    print("\nReport:")
    print(report)

    print("Prediction:", prediction)
    print("Confidence:", round(confidence * 100, 2), "%")