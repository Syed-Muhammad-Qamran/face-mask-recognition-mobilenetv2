import os
import cv2
import numpy as np
from PIL import Image
import streamlit as st
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="Face Mask Detection AI",
    page_icon="😷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #38bdf8;
        margin-bottom: 0.2rem;
    }
    .status-badge-mask {
        background-color: #065f46;
        color: #34d399;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .status-badge-nomask {
        background-color: #7f1d1d;
        color: #f87171;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Path resolution for local run vs Streamlit Cloud
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
POSSIBLE_MODEL_PATHS = [
    os.path.join(BASE_DIR, "face_mask_detection", "mask_detector.h5"),
    os.path.join(BASE_DIR, "mask_detector.h5")
]

MODEL_PATH = None
for p in POSSIBLE_MODEL_PATHS:
    if os.path.exists(p):
        MODEL_PATH = p
        break

# ---------------------------------------------------------
# Cached Model and Face Detector Loaders
# ---------------------------------------------------------
@st.cache_resource
def load_mask_model():
    if MODEL_PATH is None:
        raise FileNotFoundError("mask_detector.h5 model file was not found!")
    return load_model(MODEL_PATH)

@st.cache_resource
def load_face_detector():
    return cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

try:
    model = load_mask_model()
    face_detector = load_face_detector()
    model_ready = True
except Exception as e:
    model_ready = False
    load_err = str(e)

# ---------------------------------------------------------
# Detection Engine
# ---------------------------------------------------------
def detect_and_classify_faces(image_bgr, min_neighbors=5, scale_factor=1.1):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(
        gray, scaleFactor=scale_factor, minNeighbors=min_neighbors
    )

    detections = []
    annotated = image_bgr.copy()

    for (x, y, w, h) in faces:
        face = image_bgr[y:y+h, x:x+w]
        if face.size == 0:
            continue

        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (224, 224))
        face_resized = face_resized.astype("float32") / 255.0
        face_input = np.expand_dims(face_resized, axis=0)

        prediction = model.predict(face_input, verbose=0)[0]
        mask_prob = float(prediction[0]) * 100.0
        no_mask_prob = float(prediction[1]) * 100.0

        if mask_prob > no_mask_prob:
            label = "Mask"
            confidence = mask_prob
            box_color = (0, 220, 0) # Green in BGR
        else:
            label = "No Mask"
            confidence = no_mask_prob
            box_color = (0, 0, 220) # Red in BGR

        detections.append({
            "label": label,
            "confidence": confidence,
            "mask_prob": mask_prob,
            "no_mask_prob": no_mask_prob,
            "box": (x, y, w, h)
        })

        cv2.rectangle(annotated, (x, y), (x + w, y + h), box_color, 3)
        tag = f"{label}: {confidence:.1f}%"
        cv2.putText(
            annotated,
            tag,
            (x, max(25, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            box_color,
            2,
            cv2.LINE_AA
        )

    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    return annotated_rgb, detections

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
st.sidebar.markdown("## ⚙️ Model Settings")
st.sidebar.markdown(
    "**Architecture:** MobileNetV2 Transfer Learning\n\n"
    "**Accuracy:** ~98.7% Validation Accuracy\n\n"
    "**Input Size:** 224 × 224 RGB"
)

st.sidebar.markdown("---")
st.sidebar.subheader("Detection Sensitivity")
scale_factor = st.sidebar.slider("Face Scale Factor", 1.05, 1.40, 1.10, 0.05)
min_neighbors = st.sidebar.slider("Min Neighbors", 3, 10, 5, 1)

plot_file = os.path.join(BASE_DIR, "face_mask_detection", "accuracy_plot.png")
if os.path.exists(plot_file):
    st.sidebar.markdown("---")
    st.sidebar.image(plot_file, caption="MobileNetV2 Training Curve", use_container_width=True)

# ---------------------------------------------------------
# Main App Header
# ---------------------------------------------------------
st.markdown('<div class="main-header">😷 Face Mask Detection AI</div>', unsafe_allow_html=True)
st.markdown("Real-time AI detection powered by **MobileNetV2** and **OpenCV**. Works reliably across all browsers and devices.")

if not model_ready:
    st.error(f"Error loading model weights: {load_err}")
    st.stop()

# ---------------------------------------------------------
# Input Mode Tabs
# ---------------------------------------------------------
tab_camera, tab_upload = st.tabs([
    "📸 Live Camera Snapshot",
    "📁 Upload Image File"
])

# ---------- TAB 1: LIVE CAMERA SNAPSHOT ----------
with tab_camera:
    st.subheader("Webcam Input")
    st.markdown("Click **'Take Photo'** below to capture and analyze your face in real-time.")
    
    camera_photo = st.camera_input("Capture camera frame")
    
    if camera_photo is not None:
        pil_img = Image.open(camera_photo).convert("RGB")
        img_np = np.array(pil_img)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        with st.spinner("Analyzing frame with MobileNetV2..."):
            annotated_img, detections = detect_and_classify_faces(img_bgr, min_neighbors, scale_factor)

        col_img, col_metrics = st.columns([1.4, 1])
        with col_img:
            st.image(annotated_img, caption="Detection Output", use_container_width=True)

        with col_metrics:
            st.markdown(f"### Detected Faces: **{len(detections)}**")
            if len(detections) == 0:
                st.warning("⚠️ No face found in this frame. Ensure your face is centered and well-lit.")
            else:
                for i, d in enumerate(detections):
                    badge_class = "status-badge-mask" if d["label"] == "Mask" else "status-badge-nomask"
                    st.markdown(
                        f"""
                        <div style="background:#1e293b; padding:14px; border-radius:10px; margin-bottom:12px; border:1px solid #334155;">
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                                <strong>Face #{i+1}</strong>
                                <span class="{badge_class}">{d['label']}</span>
                            </div>
                            <div style="font-size:0.85rem; color:#94a3b8;">
                                Confidence: <strong>{d['confidence']:.1f}%</strong>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    st.write("Mask Probability:")
                    st.progress(d["mask_prob"] / 100.0)
                    st.write("No-Mask Probability:")
                    st.progress(d["no_mask_prob"] / 100.0)

# ---------- TAB 2: UPLOAD IMAGE ----------
with tab_upload:
    st.subheader("Upload an Image")
    st.markdown("Upload any JPG, JPEG, or PNG image to detect masks on one or multiple people.")
    uploaded_file = st.file_uploader("Select an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(pil_img)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        with st.spinner("Scanning faces..."):
            annotated_img, detections = detect_and_classify_faces(img_bgr, min_neighbors, scale_factor)

        col_img, col_metrics = st.columns([1.4, 1])
        with col_img:
            st.image(annotated_img, caption="Analyzed Image", use_container_width=True)

        with col_metrics:
            st.markdown(f"### Detected Faces: **{len(detections)}**")
            if len(detections) == 0:
                st.info("No faces detected in this image. You can try adjusting the sensitivity sliders in the sidebar.")
            else:
                for i, d in enumerate(detections):
                    badge_class = "status-badge-mask" if d["label"] == "Mask" else "status-badge-nomask"
                    st.markdown(
                        f"""
                        <div style="background:#1e293b; padding:14px; border-radius:10px; margin-bottom:12px; border:1px solid #334155;">
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                                <strong>Face #{i+1}</strong>
                                <span class="{badge_class}">{d['label']}</span>
                            </div>
                            <div style="font-size:0.85rem; color:#94a3b8;">
                                Confidence: <strong>{d['confidence']:.1f}%</strong>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    st.write("Mask Probability:")
                    st.progress(d["mask_prob"] / 100.0)
                    st.write("No-Mask Probability:")
                    st.progress(d["no_mask_prob"] / 100.0)
