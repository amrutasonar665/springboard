import streamlit as st
import torch
import numpy as np
from PIL import Image
import cv2
import os
import requests
from model import ResNetUNet
import albumentations as A
from albumentations.pytorch import ToTensorV2
from io import BytesIO

# --------------------------------
# CONFIGURATION
# --------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model_v2.pth")

# 🔗 UPDATE THIS LINK WITH YOUR REAL RELEASE URL
MODEL_URL = "https://github.com/amrutasonar665/springboard/releases/tag/model"

SAMPLE_IMAGE = "sample.jpg"  # must exist in repo

transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# --------------------------------
# AUTO DOWNLOAD LARGE MODEL
# --------------------------------
def download_model_if_needed():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if os.path.exists(MODEL_PATH):
        print("✔ Model found locally, skipping download")
        return

    with st.spinner("⬇ Downloading model file... please wait (only once)"):
        response = requests.get(MODEL_URL, stream=True)
        total = int(response.headers.get("content-length", 0))
        downloaded = 0

        progress = st.progress(0)

        with open(MODEL_PATH, "wb") as f:
            for data in response.iter_content(chunk_size=1024):
                if data:
                    f.write(data)
                    downloaded += len(data)
                    progress.progress(min(downloaded / total, 1.0))

        st.success("Model downloaded successfully!")

# --------------------------------
# LOAD MODEL
# --------------------------------
@st.cache_resource
def load_model():
    download_model_if_needed()

    model = ResNetUNet()
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# --------------------------------
# INFERENCE FUNCTION
# --------------------------------
def run_inference(img_source):
    image = Image.open(img_source).convert("RGB")
    img_np = np.array(image)

    augmented = transform(image=img_np)
    img_tensor = augmented["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(img_tensor)
        pred = torch.sigmoid(pred).squeeze().cpu().numpy()

    mask = (pred > 0.5).astype(np.uint8)
    mask_resized = cv2.resize(mask, (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask_3d = np.dstack([mask_resized] * 3)
    result = (img_np * mask_3d).astype(np.uint8)

    return image, result

# --------------------------------
# CUSTOM UI CSS
# --------------------------------
st.markdown("""
<style>
body {
    background-color: #0d0f1a;
    color: white;
}
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

.title-banner {
    background: linear-gradient(90deg, #00e6a8, #0077ff);
    padding: 18px;
    border-radius: 12px;
    text-align: center;
    font-size: 34px;
    color: black;
    font-weight: bold;
    letter-spacing: 2px;
    box-shadow: 0px 0px 15px #00e6a8;
}

.subtitle {
    text-align:center;
    font-size:18px;
    color:#dcdcdc;
    margin-top:-10px;
}

.image-box {
    border: 2px solid #00e6a8;
    border-radius: 14px;
    padding: 8px;
    background: rgba(255,255,255,0.05);
    box-shadow: 0px 0px 12px #00e6a8;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------
# TITLE
# --------------------------------
st.markdown("<div class='title-banner'>AI VISION EXTRACT</div>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Ultimate AI-Powered Background Removal & Object Isolation Tool</p>", unsafe_allow_html=True)

st.write("---")

# --------------------------------
# SAMPLE SECTION
# --------------------------------
st.markdown("### 📌 Example Cut-Out Preview")

try:
    sample_original, sample_output = run_inference(SAMPLE_IMAGE)
    colA, colB = st.columns(2)

    with colA:
        st.markdown("<div class='image-box'>", unsafe_allow_html=True)
        st.image(sample_original, caption="Sample Original")
        st.markdown("</div>", unsafe_allow_html=True)

    with colB:
        st.markdown("<div class='image-box'>", unsafe_allow_html=True)
        st.image(sample_output, caption="Sample Output")
        st.markdown("</div>", unsafe_allow_html=True)

except:
    st.info("⚠ Sample image not found — upload your own below.")

st.write("---")
st.markdown("## 🎯 Try It Yourself")

# --------------------------------
# UPLOAD AREA
# --------------------------------
uploaded_file = st.file_uploader(
    "Upload image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

if uploaded_file:
    img, result = run_inference(uploaded_file)

    st.write("### 🔍 Your Result")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='image-box'>", unsafe_allow_html=True)
        st.image(img, caption="Your Input Image")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='image-box'>", unsafe_allow_html=True)
        st.image(result, caption="AI Extracted Output")
        st.markdown("</div>", unsafe_allow_html=True)

    result_img = Image.fromarray(result)
    buffer = BytesIO()
    result_img.save(buffer, format="PNG")
    buffer.seek(0)

    st.download_button(
        label="📥 Download Extracted Output",
        data=buffer,
        file_name="AI_Extracted_Output.png",
        mime="image/png",
    )

else:
    st.info("Upload an image to generate high-precision AI segmentation output.")

st.write("---")
st.markdown("<p style='text-align:center; color:#888;'>Made with ❤️ using PyTorch & Streamlit | AI Vision Extract</p>", unsafe_allow_html=True)
#-----------------------------------------------
 Adding  this to download while using release
#------------------------------------------------
import os
import requests

model_path = "best_model_v2.pth"
download_url = "https://github.com/amrutasonar665/springboard/releases/download/model_file/best_model_v2.pth"

if not os.path.exists(model_path):
    with open(model_path, "wb") as f:
        print("Downloading model...")
        response = requests.get(download_url)
        f.write(response.content)
        print("Download complete.")


