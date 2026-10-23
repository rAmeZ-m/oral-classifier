
import json
import torch
import torch.nn as nn
import streamlit as st

from PIL import Image
from torchvision import models, transforms


# -------------------- Page Config --------------------
st.set_page_config(
    page_title="OralAI | Disease Classifier",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -------------------- Custom Styling --------------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0b1120 0%, #111c30 100%);
        color: #e5e7eb;
    }

    [data-testid="stSidebar"] {
        background-color: #0b1220;
        border-right: 1px solid #263449;
    }

    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #f1f5f9;
        margin-bottom: 5px;
    }

    .gradient-text {
        background: linear-gradient(90deg, #38bdf8, #2dd4bf);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .subtitle {
        color: #94a3b8;
        font-size: 16px;
        margin-bottom: 28px;
    }

    .section-card {
        background: rgba(22, 35, 56, 0.85);
        border: 1px solid #2b3b52;
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 16px;
    }

    .result-label {
        color: #94a3b8;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .result-value {
        color: #5eead4;
        font-size: 25px;
        font-weight: 700;
        margin-top: 8px;
    }

    .small-note {
        color: #94a3b8;
        font-size: 13px;
    }

    div[data-testid="stFileUploader"] {
        background: #111c30;
        border: 1px dashed #3b82a0;
        border-radius: 14px;
        padding: 12px;
    }

    div.stButton > button {
        background: linear-gradient(90deg, #0284c7, #0d9488);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
    }

    div.stButton > button:hover {
        border: 1px solid #5eead4;
        color: white;
    }

    h2, h3 {
        color: #e2e8f0;
    }

    #MainMenu, footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)


# -------------------- Load Model --------------------
@st.cache_resource
def load_model():
    with open("class_names.json", "r") as f:
        names = json.load(f)

    model = models.efficientnet_b0(weights=None)

    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.classifier[1].in_features, len(names)),
    )

    state_dict = torch.load(
        "best_oral_effnet_b0.pth",
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(state_dict)
    model.eval()

    return model, names


try:
    model, class_names = load_model()
except Exception as e:
    st.error(f"Could not load the model: {e}")
    st.stop()


# -------------------- Image Transform --------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ),
])


# -------------------- Sidebar --------------------
with st.sidebar:
    st.markdown("## 🦷 OralAI")
    st.caption("AI-powered oral image classification")

    st.divider()

    st.markdown("### About the model")
    st.write("**Architecture:** EfficientNet-B0")
    st.write("**Input size:** 224 × 224")
    st.write(f"**Classes:** {len(class_names)}")

    st.divider()

    st.markdown("### Supported classes")
    for name in class_names:
        st.write(f"• {name}")

    st.divider()

    st.markdown("### Important notice")
    st.caption(
        "This project is for educational and research purposes. "
        "Predictions are not a medical diagnosis. "
        "Consult a qualified dentist for medical concerns."
    )


# -------------------- Main Header --------------------
st.markdown("""
<div class="main-title">
    Oral<span class="gradient-text">AI</span> Classifier
</div>
<div class="subtitle">
    Intelligent oral image analysis powered by Deep Learning.
    Upload an image to explore the model's predictions.
</div>
""", unsafe_allow_html=True)

st.markdown("### Analyze an image")
st.write("Upload a clear oral photo in JPG, JPEG, or PNG format.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)


# -------------------- Prediction --------------------
if uploaded_file is None:
    st.markdown("""
    <div class="section-card">
        <h3>Ready for analysis</h3>
        <p style="color:#94a3b8;">
            Your uploaded image will appear here with the model's
            three highest-scoring predictions.
        </p>
    </div>
    """, unsafe_allow_html=True)

else:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        with st.spinner("Analyzing image..."):
            input_tensor = transform(image).unsqueeze(0)

            with torch.inference_mode():
                logits = model(input_tensor)
                probabilities = torch.softmax(logits, dim=1)[0]

            top_indices = torch.argsort(
                probabilities, descending=True
            )[:min(3, len(class_names))]

        image_col, result_col = st.columns([1, 1], gap="large")

        with image_col:
            st.markdown("### Uploaded image")
            st.image(image, use_container_width=True)

            st.caption(
                f"Image dimensions: {image.width} × {image.height}"
            )

        with result_col:
            st.markdown("### Prediction results")
            st.caption("Ranked by model output score")

            best_idx = int(top_indices[0].item())
            best_name = class_names[best_idx]
            best_score = probabilities[best_idx].item()

            st.markdown(f"""
            <div class="section-card">
                <div class="result-label">Top prediction</div>
                <div class="result-value">{best_name}</div>
                <p style="color:#cbd5e1; margin-bottom:0;">
                    Model score: {best_score:.1%}
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### Top 3 predictions")

            for rank, idx in enumerate(top_indices, start=1):
                idx = int(idx.item())
                label = class_names[idx]
                score = probabilities[idx].item()

                st.markdown(f"**{rank}. {label}**")
                st.progress(score)
                st.caption(f"Model score: {score:.1%}")

            st.info(
                "These scores reflect the model's outputs, "
                "not confirmed medical probabilities."
            )

    except Exception as e:
        st.error(f"Could not process this image: {e}")


# -------------------- Footer --------------------
st.divider()

st.markdown("""
<div style="text-align:center; color:#64748b; font-size:13px;">
    OralAI · EfficientNet-B0 · Built with PyTorch and Streamlit
</div>
""", unsafe_allow_html=True)
