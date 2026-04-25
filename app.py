import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

model = load_model("autism_model.h5")

st.title("Autism Detection System")
st.write("Upload a face image to detect Autism")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    image_pil = Image.open(uploaded_file)
    image_np = np.array(image_pil)

    img_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 
                                         'haarcascade_frontalface_default.xml')

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        face_img = img_cv[y:y+h, x:x+w]
    else:
        st.warning("No face detected, using full image")
        face_img = img_cv

    face_img = cv2.resize(face_img, (224,224))
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

    img_array = face_img / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    value = float(prediction[0][0])

    if value > 0.55:
        result = "Non-Autistic"
    elif value < 0.45:
        result = "Autistic"
    else:
        result = "Uncertain"

    st.image(face_img, caption="Processed Face", use_column_width=True)
    st.subheader("Prediction Result")
    st.success(result)
    