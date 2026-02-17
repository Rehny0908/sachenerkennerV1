import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Teachable Machine Modell laden
MODEL_PATH = "model_path_here"  # Pfad zum exportierten Teachable Machine Modell
model = tf.keras.models.load_model(MODEL_PATH)

# Titel der App
st.title("Ernstheits-Klassifikation")

# Bild hochladen
uploaded_file = st.file_uploader("Lade ein Bild hoch", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Bild anzeigen
    image = Image.open(uploaded_file)
    st.image(image, caption='Hochgeladenes Bild.', use_column_width=True)

    # Bild in das erforderliche Format umwandeln
    img_array = np.array(image.resize((224, 224))) / 255.0  # Größe je nach Teachable Machine
    img_array = np.expand_dims(img_array, axis=0)

    # Vorhersage treffen
    predictions = model.predict(img_array)
    class_names = ["Nicht ernst", "Ernst"]  # Ändere dies entsprechend deinem Modell
    predicted_class = class_names[np.argmax(predictions)]

    # Ergebnis anzeigen
    st.write(f"Das Bild wird klassifiziert als: **{predicted_class}**")
