import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

st.title("🕵️ Fake Review Detector")
st.write("This app detects whether a review is **fake** or **genuine** using machine learning.")

@st.cache_resource
def load_and_train_model():
    # Load dataset
    df = pd.read_csv("reviews-dataset.csv")

    # Display column names if error occurs
    st.write("Detected Columns:", df.columns.tolist())

    # Try auto-detecting 'text' and 'label' columns
    text_col = next((col for col in df.columns if "text" in col.lower()), None)
    label_col = next((col for col in df.columns if "label" in col.lower() or "target" in col.lower()), None)

    if not text_col or not label_col:
        raise ValueError("Could not find 'text' or 'label' columns in the dataset.")

    df.dropna(subset=[text_col, label_col], inplace=True)

    # Split and train
    X = df[text_col]
    y = df[label_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('clf', LogisticRegression())
    ])
    pipeline.fit(X_train, y_train)

    return pipeline

# Load and train
model = load_and_train_model()

# Input review
review = st.text_area("Enter a review to analyze")

if st.button("Check Review"):
    if review.strip():
        prediction = model.predict([review])[0]
        st.success(f"**Prediction:** This review is **{'Fake' if prediction == 1 else 'Genuine'}**.")
    else:
        st.warning("Please enter some text.")
