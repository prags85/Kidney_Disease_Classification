import streamlit as st
from cnnClassifier.utils.common import decodeImage
from cnnClassifier.pipeline.prediction import PredictionPipeline
import base64
import os

st.set_page_config(page_title="Kidney Disease Classifier", layout="centered")

st.title("🧠 Kidney Disease Classification")
st.write("Upload an image to get prediction.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Save the file
    with open("inputImage.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.button("🔍 Predict"):
        with st.spinner("Predicting..."):
            classifier = PredictionPipeline("inputImage.jpg")
            result = classifier.predict()  # result is a list of dict

            # ✅ Use index to access the dictionary
            prediction = result[0]["image"]
            confidence = result[0]["confidence"]

            st.success(f"🩺 Prediction: **{prediction}** (Confidence: {confidence:.2f})")
