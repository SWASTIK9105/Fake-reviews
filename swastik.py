import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from keras_preprocessing.sequence import pad_sequences
import pickle

# Load models and tokenizers
model = load_model("model/model.h5")
with open("model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
with open("model/tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

# Constants
MAX_WORDS = 300

st.title("🕵️ Fake Review Detector")
review = st.text_area("Paste a review to check if it's Fake or Real")

if st.button("Predict"):
    # Preprocess input
    tfidf_input = tfidf.transform([review]).toarray()
    seq_input = tokenizer.texts_to_sequences([review])
    padded_seq = pad_sequences(seq_input, maxlen=MAX_WORDS)

    # Prediction
    pred = model.predict([tfidf_input, padded_seq])
    prediction = "Fake Review ❌" if round(pred[0][0]) == 0 else "Genuine Review ✅"
    st.subheader(f"Prediction: {prediction}")