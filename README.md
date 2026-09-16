# MedLens

## Intelligent Medical Report Interpretation and Health Timeline

MedLens is a medical report analyzer designed to process laboratory reports and organize them into meaningful categories. The system accepts different report formats, extracts text from the uploaded file, and uses a machine learning model to identify the type of medical report.

The current implementation focuses on **Model 1: Medical Report Categorization**.

---

## Project Objective

The objective of MedLens is to simplify the processing of medical laboratory reports by automatically identifying the report category from its contents.

Instead of manually identifying the type of report, MedLens uses Natural Language Processing (NLP) and Machine Learning techniques to classify the report.

---

## Current Model

### Model 1 — Report Categorization

The first model classifies medical reports into one of 10 categories.

### Machine Learning Techniques

- TF-IDF (Term Frequency–Inverse Document Frequency)
- Logistic Regression
- Scikit-learn
- Joblib for saving the trained model

---

## Report Categories

The current model supports the following 10 categories:

1. CBC
2. Lipid Profile
3. Liver Function
4. Kidney Function
5. Thyroid
6. Diabetes HbA1c
7. Vitamin D
8. Iron Profile
9. Vitamin B12
10. Urine Routine

---

## Supported File Formats

MedLens is designed to accept:

- PDF
- DOCX
- TXT
- JPG
- JPEG
- PNG

### Text Extraction

Different file formats are processed using appropriate extraction techniques:

- **PDF** → PDF text extraction
- **DOCX** → Document text extraction
- **TXT** → Direct text reading
- **JPG/JPEG/PNG** → OCR using Tesseract

The extracted text is then passed to the report classification model.

---

## System Workflow

```text
Medical Report
      |
      v
File Processing
      |
      v
Text Extraction / OCR
      |
      v
TF-IDF Vectorization
      |
      v
Logistic Regression
      |
      v
Report Category
      |
      v
Confidence Score
