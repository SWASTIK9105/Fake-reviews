import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Title
st.title("🕵️ Fake Review Detector")
st.write("This app detects whether a review is **fake** or **genuine** using machine learning.")

@st.cache_resource
def load_and_train_model():
    # Load dataset
    df = pd.read_csv("reviews-dataset.csv")

    # Drop missing values
    df.dropna(subset=["text", "label"], inplace=True)

    # Split data
    X = df["text"]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('clf', LogisticRegression())
    ])
    pipeline.fit(X_train, y_train)

    return pipeline

# Load and train model once
model = load_and_train_model()

# Input text
review = st.text_area("Enter a review to analyze")

# Prediction
if st.button("Check Review"):
    if review.strip():
        prediction = model.predict([review])[0]
        st.success(f"**Prediction:** This review is **{'Fake' if prediction == 1 else 'Genuine'}**.")
    else:
        st.warning("Please enter some text.")

