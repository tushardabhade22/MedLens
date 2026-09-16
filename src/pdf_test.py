import joblib
from pypdf import PdfReader


def clean_text(text):
    text = text.lower()

    # Keep only useful medical keywords
    keywords = [
        "cbc",
        "hemoglobin",
        "wbc",
        "rbc",
        "platelet",
        "hematocrit",
        "mcv",
        "mch",

        "cholesterol",
        "hdl",
        "ldl",
        "triglycerides",

        "alt",
        "ast",
        "bilirubin",
        "alkaline phosphatase",
        "albumin",
        "globulin",

        "creatinine",
        "urea",
        "bun",
        "egfr",
        "uric acid",

        "tsh",
        "t3",
        "t4",
        "free t3",
        "free t4",

        "hba1c",
        "fasting glucose",
        "blood glucose",
        "ppbs",
        "fbs",

        "vitamin d",
        "25-oh vitamin d",

        "serum iron",
        "ferritin",
        "tibc",
        "uibc",
        "transferrin",

        "vitamin b12",
        "cobalamin",

        "urine",
        "specific gravity",
        "urine ph",
        "urine protein",
        "urine glucose",
        "urine rbc",
        "urine wbc"
    ]

    useful_text = []

    for keyword in keywords:
        if keyword in text:
            useful_text.append(keyword)

    return " ".join(useful_text)


# Load saved model and TF-IDF vectorizer
model = joblib.load("../models/report_classifier.pkl")
tfidf = joblib.load("../models/tfidf_vectorizer.pkl")


# Ask user for PDF path
pdf_path = input("\nEnter PDF file path: ").strip().strip('"')


# Read PDF
reader = PdfReader(pdf_path)

text = ""

for page in reader.pages:
    page_text = page.extract_text()

    if page_text:
        text += page_text + "\n"


# Check extracted text
print("\n========== EXTRACTED PDF TEXT ==========\n")
print(text)


# Clean extracted text
cleaned_text = clean_text(text)

print("\n========== CLEANED MEDICAL TEXT ==========\n")
print(cleaned_text)


# Convert cleaned text to TF-IDF
text_tfidf = tfidf.transform([cleaned_text])


# Predict report category
prediction = model.predict(text_tfidf)[0]


# Get confidence
probability = model.predict_proba(text_tfidf)[0]
confidence = probability.max()


# Display result
print("\n========== MODEL 1 RESULT ==========")
print("Predicted Report Type:", prediction)
print("Confidence:", round(confidence * 100, 2), "%")
print("====================================")