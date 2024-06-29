import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image
import joblib
import json


# Function to load and preprocess image
def load_pre_process(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = img.astype("float32") / 255.0
    return img


# Function to format class name
def format_class_name(class_name):
    return class_name.replace("_", " ").title()


# Function to predict plant disease
def predict_disease(model, image_path, class_indices):
    img = load_pre_process(image_path)
    pred = model.predict(img)
    predicted_class = np.argmax(pred, axis=1)[0]
    confidence = np.max(pred, axis=1)[0]
    predicted_name = class_indices[str(predicted_class)]
    formatted_name = format_class_name(predicted_name)
    return formatted_name, confidence


# Function to predict flower species
def predict_flower_species(model, image_path, class_indices):
    img = load_pre_process(image_path)
    pred = model.predict(img)
    predicted_class = np.argmax(pred, axis=1)[0]
    confidence = np.max(pred, axis=1)[0]
    predicted_name = class_indices[str(predicted_class)]
    formatted_name = format_class_name(predicted_name)
    return formatted_name, confidence


# Function to predict crop
def predict_crop(model, new_data, label_encoder):
    prediction = model.predict(new_data)
    return prediction


# Load plant disease detection model and class indices
model = tf.keras.models.load_model("plant_disease_model.h5")
class_indices = json.load(open("class_indices.json"))

# Load flower species classification model and class indices
flower_model = tf.keras.models.load_model("flower_detector_model.h5")
flower_class_indices = json.load(open("flower_class_indices.json"))

# Load crop recommendation model
crop_model = joblib.load("crop.pkl")

# Load label encoder for crop recommendation
label_encoder = joblib.load("label_encoder.pkl")

# Streamlit UI
st.title("\U0001F331 Plant Disease, Flower Species & Crop Recommendation")

# Add tabs for different functionalities
menu = [
    "Plant Disease Detection",
    "Flower Species Classification",
    "Crop Recommendation",
]
choice = st.sidebar.selectbox("Select Functionality", menu)

# Sidebar instructions and about section
st.sidebar.title("Instructions")
if choice == "Plant Disease Detection":
    st.sidebar.info(
        """
        1. Upload an image of a plant leaf.
        2. Wait for the model to predict the disease.
        3. Check the predicted class and confidence level.
        """
    )
elif choice == "Flower Species Classification":
    st.sidebar.info(
        """
        1. Upload an image of a flower.
        2. Wait for the model to predict the flower species.
        3. Check the predicted class and confidence level.
        """
    )
elif choice == "Crop Recommendation":
    st.sidebar.info(
        """
        1. Enter the environmental factors (N, P, K, temperature, humidity, pH, rainfall).
        2. Wait for the model to predict the recommended crop.
        """
    )

st.sidebar.title("About")
st.sidebar.info(
    """
    This app classifies plant diseases, flower species, and recommends crops using deep learning and machine learning models. 
    It has been trained on datasets of plant leaf images, flower images, and agricultural data. 
    The models' accuracy is high, but they might not be perfect.
    """
)

# Main content area based on user choice
if choice == "Plant Disease Detection" or choice == "Flower Species Classification":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("")

        if choice == "Plant Disease Detection":
            # Predict plant disease
            disease_prediction, disease_confidence = predict_disease(
                model, uploaded_file, class_indices
            )
            st.write(f"Predicted Plant Disease: **{disease_prediction}**")
            st.write(f"Disease Confidence Level: **{disease_confidence * 100:.2f}%**")

        elif choice == "Flower Species Classification":
            # Predict flower species
            flower_prediction, flower_confidence = predict_flower_species(
                flower_model, uploaded_file, flower_class_indices
            )
            st.write(f"Predicted Flower Species: **{flower_prediction}**")
            st.write(
                f"Flower Species Confidence Level: **{flower_confidence * 100:.2f}%**"
            )

elif choice == "Crop Recommendation":
    st.subheader("Enter Environmental Factors for Crop Recommendation")
    # Example new data (replace with actual new data input fields)
    new_N = st.number_input("Nitrogen (N)", min_value=0.0, max_value=500.0, value=100.0)
    new_P = st.number_input(
        "Phosphorus (P)", min_value=0.0, max_value=500.0, value=20.0
    )
    new_K = st.number_input("Potassium (K)", min_value=0.0, max_value=500.0, value=30.0)
    new_temperature = st.number_input(
        "Temperature", min_value=0.0, max_value=100.0, value=25.0
    )
    new_humidity = st.number_input(
        "Humidity", min_value=0.0, max_value=100.0, value=80.0
    )
    new_ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=6.5)
    new_rainfall = st.number_input(
        "Rainfall", min_value=0.0, max_value=1000.0, value=100.0
    )

    # Create a DataFrame from the user inputs
    new_data = pd.DataFrame(
        {
            "N": [new_N],
            "P": [new_P],
            "K": [new_K],
            "temperature": [new_temperature],
            "humidity": [new_humidity],
            "ph": [new_ph],
            "rainfall": [new_rainfall],
        }
    )

    if st.button("Predict Crop"):
        # Predict the crop
        crop_prediction = predict_crop(crop_model, new_data, label_encoder)
        crop_prediction_label = label_encoder.inverse_transform(crop_prediction)[0]
        st.write(f"Predicted Crop: **{crop_prediction_label}**")

# Feedback Form
st.write("### Feedback")
feedback = st.text_area("Provide your feedback here")
if st.button("Submit Feedback"):
    st.write("Thank you for your feedback!")
