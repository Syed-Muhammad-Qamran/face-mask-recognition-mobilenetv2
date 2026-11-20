import os
import cv2
import numpy as np
from PIL import Image
import streamlit as st
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="Face Mask Detection - MobileNetV2",
    page_icon="😷",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
# Caching Model and Face Detector
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
    model_loaded = True
except Exception as e:
    model_loaded = False
    load_error = str(e)

# ---------------------------------------------------------
# Core Detection Function
# ---------------------------------------------------------
def process_image(image_bgr, min_neighbors=5, scale_factor=1.1):
    """
    Detects faces in image_bgr, predicts Mask / No Mask for each face,
    and returns the annotated image and detection statistics.
    """
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
            color = (0, 255, 0) # Green in BGR
        else:
            label = "No Mask"
            confidence = no_mask_prob
            color = (0, 0, 255) # Red in BGR

        detections.append({
            "label": label,
            "confidence": confidence,
            "mask_prob": mask_prob,
            "no_mask_prob": no_mask_prob,
            "bbox": (x, y, w, h)
        })

        # Draw bounding box and label
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 3)
        tag = f"{label}: {confidence:.1f}%"
        cv2.putText(
            annotated,
            tag,
            (x, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA
        )

    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    return annotated_rgb, detections

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
st.sidebar.title("😷 Mask Detector")
st.sidebar.markdown(
    "Powered by **MobileNetV2** Transfer Learning trained to classify faces as **Mask** or **No Mask** (~98.7% accuracy)."
)

st.sidebar.subheader("Detection Settings")
scale_factor = st.sidebar.slider("Face Detector Scale Factor", 1.05, 1.4, 1.1, 0.05, help="Parameter specifying how much the image size is reduced at each image scale.")
min_neighbors = st.sidebar.slider("Min Neighbors", 3, 10, 5, 1, help="Higher values result in fewer detections but with higher quality.")

plot_path = os.path.join(BASE_DIR, "face_mask_detection", "accuracy_plot.png")
if os.path.exists(plot_path):
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Model Performance")
    st.sidebar.image(plot_path, caption="MobileNetV2 Training & Validation Curves", use_container_width=True)

# ---------------------------------------------------------
# Main Page
# ---------------------------------------------------------
st.title("🛡️ Real-Time Face Mask Detection System")
st.markdown(
    "Detect whether individuals are wearing face masks using **MobileNetV2** and **OpenCV**."
)

if not model_loaded:
    st.error(f"Error loading model: {load_error}")
    st.stop()

tab_upload, tab_camera, tab_live, tab_about = st.tabs([
    "📁 Upload Image",
    "📸 Webcam Snapshot",
    "🎥 Live Video Stream",
    "ℹ️ About Model"
])

# ---------- TAB 1: UPLOAD IMAGE ----------
with tab_upload:
    st.subheader("Upload an Image to Detect Masks")
    uploaded_file = st.file_uploader("Choose an image (JPG, PNG, JPEG)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        pil_image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(pil_image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        with st.spinner("Analyzing image..."):
            annotated_img, detections = process_image(image_bgr, min_neighbors, scale_factor)

        col1, col2 = st.columns([1.5, 1])
        with col1:
            st.image(annotated_img, caption="Detection Result", use_container_width=True)
            
        with col2:
            st.markdown(f"### Detected Faces: **{len(detections)}**")
            if len(detections) == 0:
                st.warning("No faces detected. Try adjusting the 'Min Neighbors' slider in the sidebar.")
            else:
                for idx, det in enumerate(detections):
                    badge = "🟢" if det["label"] == "Mask" else "🔴"
                    st.markdown(
                        f"""
                        **Face #{idx + 1}: {badge} {det['label']}**
                        * Confidence: `{det['confidence']:.2f}%`
                        * Mask Probability: `{det['mask_prob']:.2f}%`
                        * No Mask Probability: `{det['no_mask_prob']:.2f}%`
                        ---
                        """
                    )

# ---------- TAB 2: WEBCAM SNAPSHOT ----------
with tab_camera:
    st.subheader("Take a Snapshot with Your Camera")
    st.markdown("Works natively on desktop and mobile browsers without special permissions.")
    camera_file = st.camera_input("Capture Photo")

    if camera_file is not None:
        pil_image = Image.open(camera_file).convert("RGB")
        image_np = np.array(pil_image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        with st.spinner("Classifying face mask..."):
            annotated_img, detections = process_image(image_bgr, min_neighbors, scale_factor)

        col1, col2 = st.columns([1.5, 1])
        with col1:
            st.image(annotated_img, caption="Camera Analysis Result", use_container_width=True)
        with col2:
            st.markdown(f"### Detected Faces: **{len(detections)}**")
            if len(detections) == 0:
                st.info("No face detected. Please ensure your face is well-lit and facing the camera.")
            else:
                for idx, det in enumerate(detections):
                    badge = "🟢" if det["label"] == "Mask" else "🔴"
                    st.markdown(
                        f"""
                        **Face #{idx + 1}: {badge} {det['label']}**
                        * Confidence: `{det['confidence']:.2f}%`
                        * Mask Probability: `{det['mask_prob']:.2f}%`
                        * No Mask Probability: `{det['no_mask_prob']:.2f}%`
                        ---
                        """
                    )

# ---------- TAB 3: LIVE WEBRTC STREAM ----------
with tab_live:
    st.subheader("Real-Time WebRTC Video Stream")
    st.markdown("Continuous live detection using WebRTC streamer.")

    try:
        from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
        import av

        RTC_CONFIG = RTCConfiguration(
            {
                "iceServers": [
                    {"urls": ["stun:stun.l.google.com:19302"]},
                    {"urls": ["stun:stun1.l.google.com:19302"]},
                    {"urls": ["stun:stun2.l.google.com:19302"]},
                    {"urls": ["stun:stun3.l.google.com:19302"]},
                    {"urls": ["stun:stun4.l.google.com:19302"]},
                    {"urls": ["stun:stun.services.mozilla.com"]},
                    {"urls": ["stun:global.stun.twilio.com:3478"]}
                ]
            }
        )

        class LiveMaskProcessor(VideoProcessorBase):
            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")
                annotated_rgb, _ = process_image(img, min_neighbors, scale_factor)
                out_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
                return av.VideoFrame.from_ndarray(out_bgr, format="bgr24")

        webrtc_streamer(
            key="face-mask-live",
            video_processor_factory=LiveMaskProcessor,
            rtc_configuration=RTC_CONFIG,
            media_stream_constraints={"video": True, "audio": False},
        )
    except Exception as webrtc_err:
        st.warning(f"WebRTC mode note: {webrtc_err}. You can also use the 'Webcam Snapshot' tab above for immediate capture.")

# ---------- TAB 4: ABOUT MODEL ----------
with tab_about:
    st.subheader("Architecture & Technical Details")
    st.markdown("""
    ### 🔬 Pipeline Overview
    1. **Face Localization:** Haar Feature-based Cascade Classifier locates human faces in the input frame.
    2. **Preprocessing:** Extracted bounding boxes are converted from BGR to RGB, resized to $(224 \\times 224)$, normalized to range $[0, 1]$.
    3. **Transfer Learning (MobileNetV2):**
       * Base model pre-trained on ImageNet extracts spatial feature maps efficiently with depthwise separable convolutions.
       * Custom classification head: `AveragePooling2D` $\\to$ `Flatten` $\\to$ `Dense(128, ReLU)` $\\to$ `Dropout(0.5)` $\\to$ `Dense(2, Softmax)`.
    4. **Classification:** Softmax output yields probabilities for `[Mask, No Mask]`.
    
    ### 📈 Model Highlights
    * **Validation Accuracy:** $\\approx 98.7\\%$
    * **Low Latency:** Designed for lightweight real-time edge and mobile inference.
    """)
