import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Teachable Machine Modell und Labels laden
model = load_model("keras_Model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

# Titel der App
st.title("Ernstheits-Klassifikation")

# Bild hochladen
uploaded_file = st.file_uploader("Lade ein Bild hoch", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Bild anzeigen
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Hochgeladenes Bild.', use_column_width=True)

    # Bild vorbereiten
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    
    # Erstelle das Array zur Eingabe ins Modell
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image_array = np.asarray(image)
    
    # Normalisieren des Bildes
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array

    # Vorhersage treffen
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    # Ergebnis anzeigen
    st.write(f"Das Bild wird klassifiziert als: **{class_name}**")
    st.write(f"Konfidenzscore: **{confidence_score:.2f}**")
