import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import io
import os
from torch.serialization import safe_globals
from ultralytics.nn.tasks import DetectionModel
import logging

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------
# Load YOLO model safely (PyTorch 2.6+ fix)
# -----------------------
MODEL_PATH = "best.pt"

@st.cache_resource
def load_model():
    try:
        with safe_globals([DetectionModel]):
            model = YOLO(MODEL_PATH)
        logger.info("✅ Model loaded successfully from best.pt")
        return model
    except Exception as e:
        logger.error(f"❌ Failed to load model from {MODEL_PATH}. Error: {e}")
        return None

model = load_model()

# -----------------------
# Streamlit App UI
# -----------------------
st.set_page_config(page_title="NovaSight", layout="centered")
st.title("🚀 NovaSight - Real-time Object Detection")
st.markdown("Upload an image and run YOLOv8 detection.")

# Confidence slider
confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, step=0.05)

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file and model:
    try:
        # Read and convert image
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        # Inference
        results = model.predict(image_np, conf=confidence_threshold)
        result = results[0]  # Only one image

        # Draw detections
        annotated = result.plot()

        # Display image
        st.image(annotated, caption="🔍 Detection Result", use_column_width=True)

        # Display detection info
        st.subheader("📋 Detections")
        boxes = result.boxes
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
        st.error(f"❌ Error during detection: {e}")

elif not model:
    st.error("❌ Model not loaded. Please check the model path or loading logic.")
