import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
from supabase import create_client
import uuid

# ==============================
# SUPABASE SETUP
# ==============================

url = st.secrets["SUPABASE_URL"]
key = st.secrets["SUPABASE_KEY"]
supabase = create_client(url, key)

BUCKET_NAME = "uploaded_images"

# ==============================
# MODEL LOAD
# ==============================

model = load_model("keras_model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

# ==============================
# IMAGE PREPROCESSING
# ==============================

def load_image(image_file):
    image = Image.open(image_file).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    return np.expand_dims(normalized_image_array, axis=0)

# ==============================
# PREDICTION
# ==============================

def predict(image_array):
    prediction = model.predict(image_array)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = float(prediction[0][index])
    return class_name, confidence_score

# ==============================
# SUPABASE FUNCTIONS
# ==============================

def upload_file_to_supabase(file):
    unique_name = f"{uuid.uuid4()}_{file.name}"
    file_content = file.getvalue()

    supabase.storage.from_(BUCKET_NAME).upload(
        unique_name,
        file_content,
        {"content-type": file.type}
    )

    return unique_name


def save_to_supabase(class_name, confidence_score, image_path):
    data = {
        "class_name": class_name,
        "confidence_score": confidence_score,
        "image_path": image_path
    }

    supabase.table("classifications").insert(data).execute()


def fetch_uploaded_images():
    response = supabase.table("classifications").select("*").order("id", desc=True).execute()
    return response.data if response.data else []

# ==============================
# UI
# ==============================

st.title("Klassifikation von Hüten, Schuhen und Shirts")
st.write("Laden Sie ein Bild hoch, um es zu klassifizieren.")

uploaded_file = st.file_uploader("Bild auswählen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    # Upload to Supabase Storage
    image_path = upload_file_to_supabase(uploaded_file)

    # Predict
    image_array = load_image(uploaded_file)
    class_name, confidence_score = predict(image_array)

    # Save to DB
    save_to_supabase(class_name, confidence_score, image_path)

    # Show result
    color = "green" if confidence_score > 0.7 else "orange" if confidence_score > 0.5 else "red"

    st.markdown(
        f"<h3 style='color:{color};'>Vorhersage: {class_name}</h3>",
        unsafe_allow_html=True
    )
    st.write(f"Konfidenz: {confidence_score:.2f}")

# ==============================
# DISPLAY PREVIOUS IMAGES
# ==============================

st.subheader("Bereits hochgeladene Bilder")

images = fetch_uploaded_images()

if images:
    for item in images:
        public_url = supabase.storage.from_("uploaded_images").get_public_url(item["image_path"])
        st.image(
            public_url,
            caption=f"{item['class_name']} ({item['confidence_score']:.2f})"
        )
else:
    st.write("Noch keine Bilder vorhanden.")
