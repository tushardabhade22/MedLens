import pandas as pd
import random

random.seed(42)

categories = {
    "CBC": [
        "hemoglobin", "WBC", "RBC", "platelet count",
        "hematocrit", "MCV", "MCH", "MCHC",
        "neutrophils", "lymphocytes"
    ],

    "Lipid Profile": [
        "total cholesterol", "HDL cholesterol",
        "LDL cholesterol", "triglycerides",
        "VLDL", "non HDL cholesterol"
    ],

    "Liver Function": [
        "ALT", "AST", "bilirubin",
        "alkaline phosphatase", "albumin",
        "total protein", "SGOT", "SGPT",
        "globulin"
    ],

    "Kidney Function": [
        "creatinine", "urea", "BUN",
        "eGFR", "uric acid", "sodium",
        "potassium", "renal function"
    ],

    "Thyroid": [
        "TSH", "T3", "T4",
        "free T3", "free T4",
        "thyroid stimulating hormone"
    ],

    "Diabetes HbA1c": [
        "HbA1c", "hemoglobin A1c",
        "fasting blood sugar", "fasting glucose",
        "blood glucose", "postprandial glucose",
        "PPBS", "FBS", "glucose level"
    ],

    "Vitamin D": [
        "25-OH vitamin D", "vitamin D",
        "25 hydroxy vitamin D",
        "vitamin D3", "25-OH D",
        "total vitamin D"
    ],

    "Iron Profile": [
        "serum iron", "ferritin",
        "TIBC", "UIBC",
        "transferrin", "transferrin saturation",
        "iron saturation"
    ],

    "Vitamin B12": [
        "vitamin B12", "B12",
        "cobalamin", "serum B12",
        "vitamin B12 level",
        "cobalamin level"
    ],

    "Urine Routine": [
        "urine color", "urine appearance",
        "urine pH", "specific gravity",
        "urine protein", "urine glucose",
        "urine RBC", "urine WBC",
        "urine ketones", "urine bilirubin"
    ]
}


templates = [
    "{} report",
    "{} test",
    "{} laboratory report",
    "medical report showing {}",
    "patient report containing {}",
    "laboratory test with {}",
    "{} profile",
    "results for {}",
    "blood test including {}",
    "{} examination",
    "patient laboratory results for {}",
    "{} parameters were tested"
]


data = []

for label, parameters in categories.items():

    for _ in range(200):

        # Select 3-6 parameters
        number_of_parameters = random.randint(
            3,
            min(6, len(parameters))
        )

        selected = random.sample(
            parameters,
            number_of_parameters
        )

        parameter_text = " ".join(selected)

        template = random.choice(templates)

        text = template.format(parameter_text)

        data.append({
            "text": text,
            "label": label
        })


# Create DataFrame
df = pd.DataFrame(data)

# Shuffle dataset
df = df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# Save dataset
df.to_csv(
    "../data/reports.csv",
    index=False
)


print("Dataset created successfully!")

print("\nDataset shape:", df.shape)

print("\nCategory distribution:")
print(df["label"].value_counts())

print("\nFirst 10 rows:")
print(df.head(10))