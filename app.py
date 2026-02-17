import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_Model.h5", compile=False)

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
    
    # Ergebnisse anzeigen
    st.write(f"**Vorhersage:** {class_name}")
    st.write(f"**Konfidenzscore:** {confidence_score:.2f}")
