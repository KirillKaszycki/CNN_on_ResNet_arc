import streamlit as st
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
import pandas as pd
import numpy as np
from PIL import Image

model = load_model('age_regressor.h5', compile=False)


def predict_age(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    age_prediction = model.predict(img_array)[0][0]
    return age_prediction


uploaded_file = st.file_uploader("Upload the photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded photo', use_column_width=True)

    age_prediction = predict_age(uploaded_file)

    st.write(f"Perhaps, the age is: {int(age_prediction)} years")
