import streamlit as st
from PIL import Image
import numpy as np
import cv2
import logging
import torch
from torch.serialization import safe_load, safe_globals
from ultralytics.nn.tasks import DetectionModel

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------
# Streamlit App UI
# -----------------------
st.set_page_config(page_title="NovaSight", layout="centered")
st.title("🚀 NovaSight - Real-time Object Detection")
st.markdown("Upload an image and run YOLOv8 detection.")

# -----------------------
# Load YOLO model safely
# -----------------------
MODEL_PATH = "best.pt"

@st.cache_resource
def load_model():
    try:
        with safe_globals([DetectionModel]):
            model = torch.load(MODEL_PATH, weights_only=False, map_location='cpu')
            model = model.autoshape()
            logger.info("✅ Model loaded successfully.")
            return model
    except Exception as e:
        logger.error(f"❌ Failed to load model. Error: {e}")
        return None

model = load_model()

# -----------------------
# UI - Upload image
# -----------------------
confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, step=0.05)
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file and model:
    try:
        # Load image
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        # Run detection
        results = model(image_np, conf=confidence_threshold)[0]

        # Draw annotated results
        annotated = results.plot()

        # Display result
        st.image(annotated, caption="🔍 Detection Result", use_column_width=True)

        # Show detection info
        st.subheader("📋 Detections")
        boxes = results.boxes
        if boxes is not None and len(boxes) > 0:
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence_score = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                class_name = model.names[class_id]

                st.markdown(f"**{i+1}. {class_name}** — Confidence: `{confidence_score:.2f}`")
        else:
            st.info("No objects detected.")
    except Exception as e:
        st.error(f"❌ Detection error: {e}")
elif not model:
    st.error("❌ Model not loaded. Please check `best.pt`.")
