import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from keras_preprocessing.sequence import pad_sequences
import pickle
import os

# Constants
MODEL_PATH = "model/model.h5"
TOKENIZER_PATH = "model/tokenizer.pkl"
TFIDF_PATH = "model/tfidf_vectorizer.pkl"
MAX_WORDS = 300

# Title
st.title("🕵️ Fake Review Detector")

# Load model and supporting objects with error handling
@st.cache_resource
def load_artifacts():
    try:
        model = load_model(MODEL_PATH)
        with open(TOKENIZER_PATH, "rb") as f:
            tokenizer = pickle.load(f)
        with open(TFIDF_PATH, "rb") as f:
            tfidf = pickle.load(f)
        return model, tokenizer, tfidf
    except Exception as e:
        st.error(f"Error loading model or tokenizer: {e}")
        return None, None, None

model, tokenizer, tfidf = load_artifacts()

review = st.text_area("Paste a review to check if it's Fake or Real")

if st.button("Predict"):
    if model and review.strip():
        # Preprocess input
        tfidf_input = tfidf.transform([review]).toarray()
        seq_input = tokenizer.texts_to_sequences([review])
        padded_seq = pad_sequences(seq_input, maxlen=MAX_WORDS)

        # Predict
        pred = model.predict([tfidf_input, padded_seq])
        prediction = "Fake Review ❌" if round(pred[0][0]) == 0 else "Genuine Review ✅"
        st.subheader(f"Prediction: {prediction}")
    else:
        st.warning("Please make sure the model is loaded and the input is not empty.")
