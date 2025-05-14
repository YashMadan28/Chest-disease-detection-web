import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import base64

# ✅ Must be the first Streamlit command
st.set_page_config(page_title="🩻 Chest X-Ray Classifier", layout="centered")

# ✅ Background function using base64 encoding
def set_background_webp(image_path: str):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/webp;base64,{encoded_string}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-position: center;
        }}

        .block-container {{
            background-color: rgba(0, 0, 0, 0.6);  /* translucent background for content */
            padding: 2rem;
            border-radius: 10px;
        }}

        h1, h2, h3, .markdown-text-container, .stButton, .stFileUploader {{
            color: white !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ✅ Apply background
set_background_webp("back.jpg")

# ✅ Load trained model
model = tf.keras.models.load_model("chest_disease_model.keras")
class_names = ['Normal', 'Pneumonia']

# ✅ App title and instructions
st.title("🩻 Chest X-Ray Disease Detector")
st.markdown("""
Upload a **chest X-ray image** to detect whether the lungs appear **normal** or show signs of **pneumonia**.

⚠️ Please upload a **frontal chest X-ray** (PA or AP view) with no annotations or overlays.
""")

# ✅ Upload widget
uploaded_file = st.file_uploader("📁 Choose an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='🖼️ Uploaded X-ray Image', use_container_width=True)

    # Preprocess
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)

    # Display result
    st.markdown(f"## 🔍 **Prediction:** `{predicted_class}`")
    st.progress(int(confidence * 100))
    st.markdown(f"### 🧠 Confidence: `{confidence * 100:.2f}%`")

    if confidence < 0.6:
        st.warning("⚠️ This prediction has low confidence. Consider verifying with a radiologist.")
else:
    st.info("⬆️ Upload a valid chest X-ray image (JPG/PNG) to get started.")
