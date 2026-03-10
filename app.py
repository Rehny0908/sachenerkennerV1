```python
import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
from supabase import create_client
import uuid

# ==============================
# PAGE CONFIG
# ==============================

st.set_page_config(
    page_title="KI Fundbüro",
    page_icon="🔎",
    layout="wide"
)

# ==============================
# SUPABASE SETUP
# ==============================

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

BUCKET_NAME = "uploaded_images"

# ==============================
# LOAD MODEL
# ==============================

@st.cache_resource
def load_ai():
    model = load_model("keras_model.h5", compile=False)
    class_names = open("labels.txt", "r").readlines()
    return model, class_names

model, class_names = load_ai()

# ==============================
# IMAGE PREPROCESSING
# ==============================

def load_image(image_file):
    image = Image.open(image_file).convert("RGB")
    size = (224, 224)

    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    image_array = np.asarray(image)
    normalized = (image_array.astype(np.float32) / 127.5) - 1

    return np.expand_dims(normalized, axis=0)

# ==============================
# PREDICTION
# ==============================

def predict(image_array):
    prediction = model.predict(image_array)
    index = np.argmax(prediction)

    class_name = class_names[index].strip()
    confidence = float(prediction[0][index])

    return class_name, confidence

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

    response = supabase.table("classifications") \
        .select("*") \
        .order("id", desc=True) \
        .execute()

    return response.data if response.data else []


def delete_item(item_id, image_path):

    # delete database entry
    supabase.table("classifications") \
        .delete() \
        .eq("id", item_id) \
        .execute()

    # delete image from storage
    supabase.storage \
        .from_(BUCKET_NAME) \
        .remove([image_path])

# ==============================
# HEADER
# ==============================

st.title("🔎 KI Fundbüro")
st.write("Lade ein Bild hoch und lasse die KI das Objekt erkennen.")

# ==============================
# TABS
# ==============================

tab_upload, tab_search, tab_gallery = st.tabs(
    ["📤 Fundstück melden", "🔍 Suche", "🖼 Galerie"]
)

# ==============================
# TAB 1 - UPLOAD
# ==============================

with tab_upload:

    st.subheader("Bild hochladen")

    uploaded_file = st.file_uploader(
        "Wähle ein Bild",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:

        col1, col2 = st.columns(2)

        with col1:
            st.image(uploaded_file, caption="Vorschau", use_container_width=True)

        with col2:

            if st.button("KI Klassifikation starten"):

                with st.spinner("Analysiere Bild..."):

                    image_path = upload_file_to_supabase(uploaded_file)

                    image_array = load_image(uploaded_file)
                    class_name, confidence = predict(image_array)

                    save_to_supabase(class_name, confidence, image_path)

                st.success("Analyse abgeschlossen")

                st.metric("Erkannte Klasse", class_name)
                st.progress(confidence)
                st.write(f"Konfidenz: **{confidence:.2%}**")

# ==============================
# TAB 2 - SEARCH
# ==============================

with tab_search:

    st.subheader("Fundstücke durchsuchen")

    images = fetch_uploaded_images()

    classes = list(set([item["class_name"] for item in images])) if images else []

    selected_class = st.selectbox(
        "Nach Klasse filtern",
        ["Alle"] + classes
    )

    filtered = images

    if selected_class != "Alle":
        filtered = [i for i in images if i["class_name"] == selected_class]

    st.write(f"{len(filtered)} Ergebnisse gefunden")

    cols = st.columns(3)

    for i, item in enumerate(filtered):

        public_url = supabase.storage \
            .from_(BUCKET_NAME) \
            .get_public_url(item["image_path"])

        with cols[i % 3]:

            st.image(public_url, use_container_width=True)

            st.caption(
                f"{item['class_name']} ({item['confidence_score']:.2f})"
            )

# ==============================
# TAB 3 - GALLERY
# ==============================

with tab_gallery:

    st.subheader("Alle Fundstücke")

    images = fetch_uploaded_images()

    if not images:
        st.info("Noch keine Bilder vorhanden")

    else:

        cols = st.columns(3)

        for i, item in enumerate(images):

            public_url = supabase.storage \
                .from_(BUCKET_NAME) \
                .get_public_url(item["image_path"])

            with cols[i % 3]:

                st.image(public_url, use_container_width=True)

                st.markdown(
                    f"""
                    **{item['class_name']}**  
                    Konfidenz: {item['confidence_score']:.2f}
                    """
                )

                if st.button("✅ Abgeholt", key=f"delete_{item['id']}"):

                    delete_item(item["id"], item["image_path"])

                    st.success("Fundstück wurde abgeholt")

                    st.rerun()
```
