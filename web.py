import numpy as np
import pandas as pd
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st


word_index = imdb.get_word_index()
reverse_word_index = {value:key for key, value in word_index.items()}

model = load_model('RNN.h5')

# function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# processing user input
def process_user_input(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen = 500)
    return padded_review

# predict function
def predictions(reviews):
    preprocessed_input = process_user_input(reviews)
    # print(preprocessed_input.shape)
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment, prediction[0][0]

# app
st.set_page_config(page_title="Sentiment Analysis", layout="centered", page_icon="🎬")
st.markdown(
    "<h1 style='text-align: center; color: #00adb5;'>🎬 Sentiment Analysis</h1>",
    unsafe_allow_html=True
)
st.markdown("<p style='text-align: center;'>Enter a movie review and see if it's positive or negative!</p>", unsafe_allow_html=True)

# Input
user_input = st.text_area("✏️ Enter your movie review below:", height=150, placeholder="E.g., This movie was absolutely amazing!")

# Predict button
if st.button("🔍 Predict the sentiment"):
    if not user_input.strip():
        st.warning("⚠️ Please enter a movie review.")
    else:
        preprocessed = process_user_input(user_input)
        prediction = model.predict(preprocessed)[0][0]
        # sentiment = 'Positive ' if prediction > 0.5 else 'Negative '

        st.subheader("📊 Prediction Results")
        # st.markdown(f"**Sentiment:** `{sentiment}`")
        st.progress(prediction if prediction > 0.5 else 1 - prediction)
        st.markdown(f"**Confidence Score:** `{prediction:.2f}`")

        # Optional: Give text feedback
        if prediction > 0.85:
            st.success("Looks like a strong positive review! 😄")
        elif prediction < 0.15:
            st.error("That seems like a negative review. 😞")
        else:
            st.info("Hmm, that's a bit neutral or unclear. 😐")
else:
    st.markdown("<p style='color:gray;'>👆 Enter your review and hit the button to analyze the sentiment.</p>", unsafe_allow_html=True)