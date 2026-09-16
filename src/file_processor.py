import os
import joblib

from pypdf import PdfReader
from docx import Document
from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# =========================================================
# LOAD MODEL 1
# =========================================================

model = joblib.load("../models/report_classifier.pkl")
tfidf = joblib.load("../models/tfidf_vectorizer.pkl")


# =========================================================
# FILE EXTRACTION FUNCTIONS
# =========================================================

def extract_from_pdf(file_path):
    text = ""

    reader = PdfReader(file_path)

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


def extract_from_docx(file_path):
    document = Document(file_path)

    text = ""

    for paragraph in document.paragraphs:
        text += paragraph.text + "\n"

    return text


def extract_from_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def extract_from_image(file_path):
    image = Image.open(file_path)

    text = pytesseract.image_to_string(image)

    return text


# =========================================================
# DETECT FILE TYPE AND EXTRACT TEXT
# =========================================================

def extract_text(file_path):

    extension = os.path.splitext(file_path)[1].lower()

    if extension == ".pdf":
        return extract_from_pdf(file_path)

    elif extension == ".docx":
        return extract_from_docx(file_path)

    elif extension == ".txt":
        return extract_from_txt(file_path)

    elif extension in [".jpg", ".jpeg", ".png"]:
        return extract_from_image(file_path)

    else:
        raise ValueError(
            "Unsupported file format. "
            "Supported formats: PDF, DOCX, TXT, JPG, JPEG, PNG"
        )


# =========================================================
# MODEL 1 PREDICTION
# =========================================================

def predict_report_type(text):

    # Convert extracted text into TF-IDF features
    text_tfidf = tfidf.transform([text])

    # Predict report category
    prediction = model.predict(text_tfidf)[0]

    # Get prediction probabilities
    probability = model.predict_proba(text_tfidf)[0]

    # Highest probability
    confidence = probability.max()

    return prediction, confidence


# =========================================================
# MAIN PROGRAM
# =========================================================

if __name__ == "__main__":

    file_path = input("Enter file path: ").strip().strip('"')

    try:

        # -------------------------------------------------
        # STEP 1: Extract text
        # -------------------------------------------------

        text = extract_text(file_path)

        print("\n========== EXTRACTED TEXT ==========\n")
        print(text)

        print("\n====================================")
        print("Text extraction completed successfully.")


        # -------------------------------------------------
        # STEP 2: Model 1 prediction
        # -------------------------------------------------

        prediction, confidence = predict_report_type(text)

        print("\n========== MODEL 1 RESULT ==========")

        print("Predicted Report Type:", prediction)

        print("Confidence:", round(confidence * 100, 2), "%")

        print("====================================")


    except Exception as e:

        print("\nError:", e)