import streamlit as st
import joblib
from utils import clean_text

model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
label_map = {0: "😡 Negative", 1: "😐 Neutral", 2: "😊 Positive"}

st.title("📝 Product Review Sentiment Analyzer")

review = st.text_area("Enter Review Text:")
if st.button("Analyze"):
    if review.strip() == "":
        st.warning("Please enter text.")
    else:
        cleaned = clean_text(review)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]
        st.success(f"**Predicted Sentiment:** {label_map[prediction]}")
