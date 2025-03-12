import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained model
model = load_model("C:/Users/ASHMITA/disease_detection/disease_prediction.h5")

# Define function for preprocessing the uploaded image
def preprocess_image(img):
    img = img.resize((224, 224))  # Resize to match the model input size
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Define function to apply grayscale, threshold, and edge detection
def preprocess_visual(img):
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(gray, 100, 200)
    return gray, threshold, edges

# Streamlit UI
st.set_page_config(page_title="Plant Disease Detection", layout="wide")

# Header
st.markdown("<h1 style='text-align: center; color: green;'>🌿 Plant Disease Detection</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Upload an image of a plant leaf to check if it is healthy or diseased.</h3>", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("Upload a plant leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # Preprocess and predict
    img = Image.open(uploaded_file)
    processed_img = preprocess_image(img)
    prediction = model.predict(processed_img)
    label = "Healthy Leaf" if prediction[0][0] > 0.5 else "Diseased Leaf"

    # Display the prediction
    st.markdown(f"<h1 style='color: blue;'>Prediction: {label}</h1>", unsafe_allow_html=True)

     # Provide solutions based on prediction
    if label == "Healthy Leaf":
        st.success("The leaf is healthy! Keep up the good work.")
        st.info(" **Solution:** Ensure the plant receives adequate water and sunlight. Check for pests regularly to maintain its health.")
    else:
        st.error("The plant is diseased! Immediate action is required.")
        st.warning("**Solution:** Apply appropriate fungicides or pesticides. Remove infected leaves to prevent the disease from spreading. Ensure proper soil drainage and avoid overwatering.")


    # Preprocessing visualization
    st.markdown("<h1 style='color: green;'>Preprocessed Images</h1>", unsafe_allow_html=True)
    gray, threshold, edges = preprocess_visual(img)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(gray, caption="Grayscale", use_container_width=True, channels="GRAY")
    with col2:
        st.image(threshold, caption="Thresholded", use_container_width=True, channels="GRAY")
    with col3:
        st.image(edges, caption="Edges Detected", use_container_width=True, channels="GRAY")

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>© 2024 Plant Disease Detection System</p>", unsafe_allow_html=True)
