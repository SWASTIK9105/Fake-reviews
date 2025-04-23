import streamlit as st
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

st.title("🕵️ Fake Review Detector (with Smart Heuristics)")
st.write("This app detects whether a review is **fake** or **genuine** using both machine learning and smart rules.")

# Heuristic-based flagging
def heuristic_fake_score(text):
    text = text.lower()
    score = 0

    # Too many superlatives
    superlatives = ["amazing", "incredible", "perfect", "best", "awesome", "fantastic", "10 stars", "love it", "must buy"]
    score += sum(1 for word in superlatives if word in text)

    # Too many exclamation marks
    score += text.count("!") // 2

    return score

@st.cache_resource
def load_and_train_model():
    df = pd.read_csv("reviews-dataset.csv")

    st.write("Detected Columns:", df.columns.tolist())

    # Auto-detect columns
    text_col = next((col for col in df.columns if "text" in col.lower()), None)
    label_col = next((col for col in df.columns if "label" in col.lower() or "target" in col.lower()), None)

    if not text_col or not label_col:
        raise ValueError("Could not find 'text' or 'label' columns in the dataset.")

    df.dropna(subset=[text_col, label_col], inplace=True)

    X = df[text_col]
    y = df[label_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('clf', LogisticRegression())
    ])
    pipeline.fit(X_train, y_train)

    return pipeline

# Load model
model = load_and_train_model()

# User input
review = st.text_area("Enter a review to analyze")

if st.button("Check Review"):
    if review.strip():
        ml_pred = model.predict([review])[0]
        score = heuristic_fake_score(review)

        # Combine ML and heuristic (override if suspicious enough)
        if score >= 3:
            st.warning("⚠️ This review seems suspiciously enthusiastic.")
            st.success("**Prediction (Adjusted):** This review is **Fake** (based on content patterns).")
        else:
            label = "Fake" if ml_pred == 1 else "Genuine"
            st.success(f"**Prediction:** This review is **{label}**.")
    else:
        st.warning("Please enter some text.")
