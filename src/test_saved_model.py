import joblib

# Load saved model
model = joblib.load("../models/report_classifier.pkl")
tfidf = joblib.load("../models/tfidf_vectorizer.pkl")

print("Medical Report Classifier")
print("Type 'exit' to stop.")

while True:

    # Take report from user
    report = input("\nEnter medical report text: ")

    if report.lower() == "exit":
        break

    # Convert report to TF-IDF
    report_tfidf = tfidf.transform([report])

    # Predict category
    prediction = model.predict(report_tfidf)[0]

    # Get confidence
    probability = model.predict_proba(report_tfidf)[0]
    confidence = probability.max()

    print("\nPredicted Report Type:", prediction)
    print("Confidence:", round(confidence * 100, 2), "%")