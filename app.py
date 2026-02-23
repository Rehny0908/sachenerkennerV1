import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
from supabase import create_client, Client

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Supabase Konfiguration
url = "DEINE_SUPABASE_URL"
key = "DEIN_SUPABASE_ANON_KEY"
supabase: Client = create_client(url, key)

# Load the model
model = load_model("keras_model.h5", compile=False)

# Load the labels
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
    data = {"class_name": class_name, "confidence_score": confidence_score}
    supabase.table("classifications").insert(data).execute()

st.title("Klassifikation von Hüten, Schuhen und Shirts")
st.write("Laden Sie ein Bild hoch, um zu sehen, was es ist!")

# Bild-Upload-Funktion
uploaded_file = st.file_uploader("Wählen Sie ein Bild aus...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Bild anzeigen
    st.image(uploaded_file, caption="Hochgeladenes Bild.", use_column_width=True)
    
    # Bildverarbeitung
    image_array = load_image(uploaded_file)
    
    # Vorhersage
    class_name, confidence_score = predict(image_array)

    # Ergebnisse speichern
    save_to_supabase(class_name, confidence_score)

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
