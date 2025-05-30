import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os
print(os.listdir())


# Load data
df = pd.read_csv("data/spam.csv", encoding="latin-1")[["v1", "v2"]]
df.columns = ["label", "text"]

# Preprocess
X = df["text"]
y = df["label"].map({"ham": 0, "spam": 1})

# Vectorize
vectorizer = TfidfVectorizer()
vectorizer.fit(X)
joblib.dump(vectorizer, "models/vectorizer.pkl")
X_vec = vectorizer.transform(X)
# Train
model = LogisticRegression()
model.fit(X_vec, y)

# Save model + vectorizer
joblib.dump(model, "models/model.pkl")
