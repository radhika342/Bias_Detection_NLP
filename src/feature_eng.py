import os
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer


DATA_PATH = "data/raw/bias_dataset.csv"
PROCESSED_DIR = "data/processed"

os.makedirs(PROCESSED_DIR, exist_ok=True)


df = pd.read_csv(DATA_PATH)

texts = df["text"].tolist()
labels = df["label"].values

print("Dataset Loaded:", len(df))
print("Label Distribution:\n", df["label"].value_counts())


# TF-IDF Features
print("\nGenerating TF-IDF features...")

tfidf = TfidfVectorizer(
    max_features=3000,
    ngram_range=(1, 2),
    stop_words="english"
)

X_tfidf = tfidf.fit_transform(texts)


np.save(os.path.join(PROCESSED_DIR, "X_tfidf.npy"), X_tfidf.toarray())
np.save(os.path.join(PROCESSED_DIR, "y.npy"), labels)

print("TF-IDF shape:", X_tfidf.shape)


# SBERT Embeddings
print("\nGenerating SBERT embeddings...")

model = SentenceTransformer("all-MiniLM-L6-v2")

X_sbert = model.encode(texts, show_progress_bar=True)

np.save(os.path.join(PROCESSED_DIR, "X_sbert.npy"), X_sbert)

print("SBERT shape:", X_sbert.shape)

print("\nFeature extraction complete.")
