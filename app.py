import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
from supabase import create_client, Client
import uuid
import requests
from io import BytesIO

# Supabase-Client initialisieren
url = st.secrets["SUPABASE_URL"]
key = st.secrets["SUPABASE_KEY"]
supabase: Client = create_client(url, key)

# Laden des Modells
model = load_model("keras_model.h5", compile=False)

# Laden der Labels
class_names = open("labels.txt", "r").readlines()

def load_image(image_path):
    """Lädt und bearbeitet das Bild für die Vorhersage."""
    image = Image.open(image_path).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    return np.expand_dims(normalized_image_array, axis=0)

def predict(image_array):
    """Gibt die Vorhersage des Modells für das Bild zurück."""
    prediction = model.predict(image_array)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]
    return class_name, confidence_score

def save_to_supabase(class_name, confidence_score):
    """Speichert die Vorhersage in der Supabase-Datenbank."""
    data = {
        "class_name": str(class_name),
        "confidence_score": float(confidence_score)
    }
    supabase.table("classifications").insert(data).execute()

def upload_file_to_supabase(file):
    """Lädt ein Bild in den Supabase Bucket hoch."""
    bucket_name = "uploaded_images"
    
    unique_name = f"{uuid.uuid4()}_{file.name}"
    file_content = file.getvalue()

    response = supabase.storage.from_(bucket_name).upload(
        path=unique_name,
        file=file_content,
        file_options={"upsert": "true"}
    )

    return unique_name  # Rückgabe des einzigartigen Namens

def fetch_uploaded_images():
    """Lädt bereits hochgeladene Bilder aus der Supabase-Datenbank."""
    response = supabase.table("classifications").select("class_name, created_at").execute()
    return response.data

def display_uploaded_images():
    """Zeigt die hochgeladenen Bilder an."""
    images = fetch_uploaded_images()
    bucket_url = "https://your-supabase-project-url.supabase.co/storage/v1/object/public/uploaded_images/"  # Ersetze durch deine URL

    if images:
        for item in images:
            st.image(f"{bucket_url}{item['class_name']}", caption=f"{item['class_name']}, hochgeladen am {item['created_at']}")
    else:
        st.write("Keine hochgeladenen Bilder gefunden.")

st.title("Klassifikation von Hüten, Schuhen und Shirts")
st.write("Laden Sie ein Bild hoch, um zu sehen, was es ist!")

# Bild-Upload-Funktion
uploaded_file = st.file_uploader("Wählen Sie ein Bild aus...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Lade das Bild in den Supabase Bucket hoch
    unique_name = upload_file_to_supabase(uploaded_file)

    # Bildverarbeitung
    image_array = load_image(uploaded_file)
    class_name, confidence_score = predict(image_array)

    # Ergebnisse speichern
    save_to_supabase(unique_name, confidence_score)

    # Dynamisches Farb-Tag für das Ergebnis
    if confidence_score > 0.7:
        color = "green"
    elif confidence_score > 0.5:
        color = "yellow"
    else:
        color = "red"

    # Ergebnisse anzeigen
    st.markdown(f"<h3 style='color:{color};'>**Vorhersage:** {class_name}</h3>", unsafe_allow_html=True)
    st.write(f"**Konfidenzscore:** {confidence_score:.2f}")

# Bereits hochgeladene Bilder anzeigen
st.subheader("Bereits hochgeladene Bilder")
display_uploaded_images()
