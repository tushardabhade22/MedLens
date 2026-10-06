
import joblib

# Load combined MedLens model
model = joblib.load("../models/medlens_model.pkl")

print("Medical Report Classifier")
print("Type 'exit' to stop.")

while True:

    # Take report from user
    report = input("\nEnter medical report text: ")

    if report.lower() == "exit":
        break

    # Predict category
    prediction = model.predict([report])[0]

    # Get confidence
    probability = model.predict_proba([report])[0]
    confidence = probability.max()

    print("\nPredicted Report Type:", prediction)
    print("Confidence:", round(confidence * 100, 2), "%")
