import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# -----------------------------
# PAGE SETTINGS
# -----------------------------
st.set_page_config(
    page_title="Crop Disease Detection",
    page_icon="🌱",
    layout="centered"
)

st.title("🌱 Crop Disease Detection System")
st.write(
    "Upload a crop leaf image to predict its possible disease "
    "using our trained AI model."
)

# -----------------------------
# MODEL PATHS
# -----------------------------
MODEL_PATH = "crop_disease_model.keras"

CLASS_PATH = "class_names.txt"

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_class_names():
    with open(CLASS_PATH, "r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]

# -----------------------------
# CHECK FILES
# -----------------------------
if not os.path.exists(MODEL_PATH):
    st.error("Trained model not found. Please check the models folder.")
    st.stop()

if not os.path.exists(CLASS_PATH):
    st.error("Class names file not found.")
    st.stop()

# -----------------------------
# IMAGE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload a crop leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Leaf Image",
            use_container_width=True
        )

        if st.button("🔍 Detect Disease"):

            with st.spinner("Analyzing leaf image..."):

                model = load_model()
                class_names = load_class_names()

                image = image.resize((224, 224))

                image_array = np.array(
                    image,
                    dtype=np.float32
                )

                image_array = np.expand_dims(
                    image_array,
                    axis=0
                )

                predictions = model.predict(
                    image_array,
                    verbose=0
                )

                predicted_index = int(
                    np.argmax(predictions[0])
                )

                confidence = float(
                    predictions[0][predicted_index]
                ) * 100

                if predicted_index >= len(class_names):
                    st.error("Prediction class mismatch.")
                    st.stop()

                predicted_class = class_names[predicted_index]

            st.success("Analysis completed!")

            st.subheader("Prediction Result")

            st.write("**Predicted class:**", predicted_class)

            st.metric(
                "Model Confidence",
                f"{confidence:.2f}%"
            )

            st.info(
                "This result is an AI prediction, not a confirmed "
                "agricultural diagnosis. Results may be less reliable "
                "for images unlike the training dataset."
            )

    except Exception as error:
        st.error(f"Unable to analyze the image: {error}")
