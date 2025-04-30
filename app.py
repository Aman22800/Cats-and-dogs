import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Page config
st.set_page_config(page_title="Cat vs Dog Classifier", layout="centered")

# Title
st.title("🐱🐶 Cat vs Dog Classifier")
st.markdown("Upload an image of a cat or dog and the model will predict the animal.")

# Load trained model
@st.cache_resource
def load_cat_dog_model():
    return load_model("model/cat_dog_model.h5")

model = load_cat_dog_model()

# Function to preprocess image
def preprocess_image(uploaded_image):
    image = uploaded_image.resize((256, 256))  # Resize to match model input
    img_array = np.array(image) / 255.0  # Normalize
    img_array = img_array.reshape((1, 256, 256, 3))  # Add batch dimension
    return img_array

# Function to predict
def predict_image(image_array):
    prediction = model.predict(image_array)
    if prediction[0][0] < 0.5:
        return "🐱 Cat"
    else:
        return "🐶 Dog"

# Upload interface
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess and Predict
    image_array = preprocess_image(image)
    label = predict_image(image_array)

    st.success(f"Prediction: **{label}**")
