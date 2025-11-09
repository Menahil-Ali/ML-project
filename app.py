import streamlit as st
import pickle

st.title("Sentiment Analysis App")

text = st.text_input("Enter your review:")

if st.button("Analyze"):
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    x = vectorizer.transform([text])
    result = model.predict(x)[0]
    st.write("Prediction:", "Positive" if result == 1 else "Negative")
