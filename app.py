import json
import torch
import torch.nn as nn
import streamlit as st
from PIL import Image
from torchvision import models, transforms

st.set_page_config(page_title="Oral Disease Classifier")

@st.cache_resource
def load_model():
    with open("class_names.json") as f:
        names = json.load(f)
    m = models.efficientnet_b0(weights=None)
    m.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(m.classifier[1].in_features, len(names)),
    )
    m.load_state_dict(torch.load("best_oral_effnet_b0.pth", map_location="cpu"))
    m.eval()
    return m, names

model, class_names = load_model()

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

st.title("Oral Disease Classifier")
st.caption("EfficientNet-B0 on 6 oral conditions. Educational use only, not a medical diagnosis.")

file = st.file_uploader("Upload an oral photo", type=["jpg", "jpeg", "png"])
if file:
    img = Image.open(file).convert("RGB")
    st.image(img, use_container_width=True)
    with torch.no_grad():
        probs = model(tf(img).unsqueeze(0)).softmax(1)[0]
    for i in probs.argsort(descending=True)[:3]:
        st.write(f"**{class_names[i]}**: {probs[i]:.1%}")
        st.progress(float(probs[i]))