import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset
df = pd.read_csv("../data/reports.csv")

X = df["text"]
y = df["label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Create TF-IDF
tfidf = TfidfVectorizer()

# Learn vocabulary from training data
X_train_tfidf = tfidf.fit_transform(X_train)

# Transform test data using the same vocabulary
X_test_tfidf = tfidf.transform(X_test)

print("Original training samples:", X_train.shape)
print("TF-IDF training shape:", X_train_tfidf.shape)

print("\nOriginal testing samples:", X_test.shape)
print("TF-IDF testing shape:", X_test_tfidf.shape)

print("\nNumber of features:", len(tfidf.get_feature_names_out()))

print("\nFirst 20 features:")
print(tfidf.get_feature_names_out()[:20])