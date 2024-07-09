import streamlit as st
import numpy as np
import joblib
from PIL import Image
from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt


def load_image(file, dimension=(104, 104)):
    img = imread(file)
    img_resized = resize(img, dimension, anti_aliasing=True, mode='reflect')
    flat_data = [img_resized.flatten()]
    return img, flat_data


# Load the model and target names
model_filename = "rice_disease_model.pkl"
clf = joblib.load(model_filename)
target_names = np.load("target_names.npy")

st.title("Rice Disease Prediction")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Predict
    plot_img, flat_img = load_image(uploaded_file)
    prediction = clf.predict(flat_img)
    predicted_disease = target_names[prediction[0]]

    st.write(f"Predicted Disease is: {predicted_disease}")
