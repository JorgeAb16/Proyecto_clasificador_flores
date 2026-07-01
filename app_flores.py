import json
from pathlib import Path
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(
    page_title="Clasificador de Flores",
    page_icon="🌸",
    layout="centered"
)

# ── Estilos ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #1a0533, #3b1060, #1a2a1a);
    color: #f0f0f0;
}

.hero {
    text-align: center;
    padding: 2rem 1rem 1rem;
}
.hero h1 {
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(90deg, #f953c6, #b91d73);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.hero p {
    color: #c9b0d0;
    font-size: 0.95rem;
    margin-top: 0;
}

.badge {
    display: inline-block;
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 999px;
    padding: 0.3rem 1rem;
    font-size: 0.8rem;
    color: #e0c8f0;
    margin-bottom: 1.5rem;
}

.upload-box {
    background: rgba(255,255,255,0.05);
    border: 2px dashed rgba(249,83,198,0.5);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    margin-bottom: 1.5rem;
}

.result-card {
    background: rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.1);
    margin-top: 1.5rem;
}
.result-emoji {
    font-size: 4rem;
    margin-bottom: 0.3rem;
}
.result-label {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #f953c6, #b91d73);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.footer {
    text-align: center;
    color: #5c4a6a;
    font-size: 0.78rem;
    margin-top: 3rem;
    padding-bottom: 1rem;
}

#MainMenu, footer, header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Constantes ────────────────────────────────────────────────────────────────
IMG_SIZE    = (224, 224)
MODEL_PATHS = [Path("modelo_flores.keras"), Path("modelo_flores.h5")]
CLASS_PATH  = Path("clases.json")

LABELS_ES = {
    "daisy":      "Margarita",
    "dandelion":  "Diente de León",
    "rose":       "Rosa",
    "sunflower":  "Girasol",
    "tulip":      "Tulipán",
}
EMOJIS = {
    "daisy":      "🌼",
    "dandelion":  "🌿",
    "rose":       "🌹",
    "sunflower":  "🌻",
    "tulip":      "🌷",
}

# ── Carga ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def cargar_modelo():
    for path in MODEL_PATHS:
        if path.exists():
            return tf.keras.models.load_model(path, compile=False)
    st.error("No se encontró el modelo. Coloca modelo_flores.keras junto a app.py.")
    st.stop()

@st.cache_data
def cargar_clases():
    if CLASS_PATH.exists():
        with open(CLASS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return ["daisy", "dandelion", "rose", "sunflower", "tulip"]

def preparar_imagen(img):
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

def predecir(img):
    preds = modelo.predict(preparar_imagen(img), verbose=0)[0]
    return [
        (clases[i], LABELS_ES.get(clases[i], clases[i]), float(preds[i]) * 100)
        for i in np.argsort(preds)[::-1]
    ]

modelo = cargar_modelo()
clases = cargar_clases()

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🌸 Clasificador de Flores</h1>
    <p>Clasificador de imágenes con Transfer Learning — MobileNetV2</p>
</div>
<div style="text-align:center">
    <span class="badge">👤 Jorge Abraham Fajardo López · 20231900189</span>
</div>
""", unsafe_allow_html=True)

# ── Upload ────────────────────────────────────────────────────────────────────
archivo = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if not archivo:
    st.markdown("""
    <div class="upload-box">
        <p style="font-size:2rem; margin:0">📂</p>
        <p style="color:#c9b0d0; margin:0.3rem 0 0">Arrastra una imagen aquí o usa el botón de arriba</p>
        <p style="color:#5c4a6a; font-size:0.8rem; margin:0.2rem 0 0">Formatos: JPG · JPEG · PNG</p>
    </div>
    """, unsafe_allow_html=True)

else:
    imagen     = Image.open(archivo)
    resultados = predecir(imagen)

    # Obtenemos solo el primer resultado (el más alto)
    top_key   = resultados[0][0]
    top_label = resultados[0][1]
    emoji     = EMOJIS.get(top_key, "🌸")

    st.image(imagen, caption="Imagen analizada", use_container_width=True)

    # ── Texto agregado antes de la tarjeta ───────────────────────────────────
    st.markdown("""
    <div style="text-align: center; margin-top: 1.5rem; margin-bottom: -0.5rem;">
        <span style="font-size: 0.85rem; font-weight: 600; letter-spacing: 0.15em; text-transform: uppercase; color: #c9b0d0;">
            Resultado del análisis:
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Tarjeta de resultado limpia: Solo muestra el emoji y el nombre de la flor
    st.markdown(f"""
    <div class="result-card" style="margin-top: 0.5rem;">
        <div class="result-emoji">{emoji}</div>
        <div class="result-label">{top_label}</div>
    </div>
    """, unsafe_allow_html=True)