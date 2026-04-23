import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

MODEL_PATH = "model.pth"

CLASS_NAMES = [
    "Apple", "Banana", "Cherry", "Chickoo",
    "Grapes", "Kiwi", "Mango", "Orange", "Strawberry",
]

TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
])

@st.cache_resource
def load_model():
    device = torch.device("cpu")
    model = torch.jit.load(MODEL_PATH, map_location=device)
    model.eval()
    return model, device

def predict(image: Image.Image, model, device):
    tensor = TRANSFORM(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
    top_idx = int(np.argmax(probs))
    return CLASS_NAMES[top_idx], probs

st.set_page_config(page_title="Clasificador de Frutas", page_icon="🍎", layout="centered")
st.title("Clasificador de Frutas")
st.caption("Toma una foto con tu cámara o sube una imagen para identificar la fruta.")

try:
    model, device = load_model()
except Exception as e:
    st.error(f"No se pudo cargar el modelo: {e}")
    st.stop()

source = st.radio("Fuente de imagen", ["Cámara", "Subir archivo"], horizontal=True)

image = None

if source == "Cámara":
    photo = st.camera_input("Apunta la cámara hacia una fruta y toma la foto")
    if photo:
        image = Image.open(photo)
else:
    uploaded = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png", "webp"])
    if uploaded:
        image = Image.open(uploaded)

if image is not None:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Imagen de entrada", use_container_width=True)

    with col2:
        with st.spinner("Clasificando…"):
            label, probs = predict(image, model, device)

        st.success(f"**Fruta detectada:** {label}")
        st.metric("Confianza", f"{probs.max() * 100:.1f}%")

    st.subheader("Probabilidades por clase")
    prob_dict = {name: float(p) for name, p in zip(CLASS_NAMES, probs)}
    sorted_probs = dict(sorted(prob_dict.items(), key=lambda x: x[1], reverse=True))
    st.bar_chart(sorted_probs)
