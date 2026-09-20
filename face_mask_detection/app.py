import os
import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

# Resolve paths relative to this script's own folder, not the process's
# working directory (Streamlit Cloud runs from the repo root, not this
# subfolder, so a plain relative filename can fail to be found).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "mask_detector.h5")

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Face Mask Detection System", layout="wide")

st.title("😷 Face Mask Detection System")
st.write("Real-time face mask detection using your webcam.")

# -----------------------------
# Load Model (cached so it only loads once)
# -----------------------------
@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)

model = get_model()

class_names = ["Mask", "No Mask"]

# -----------------------------
# Face Detector (cached)
# -----------------------------
@st.cache_resource
def get_face_detector():
    return cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

face_detector = get_face_detector()

# -----------------------------
# Sidebar status placeholder
# -----------------------------
status_placeholder = st.sidebar.empty()
status_placeholder.info("Waiting for webcam feed...")

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# -----------------------------
# Video Processor
# -----------------------------
class MaskDetector(VideoProcessorBase):
    def __init__(self):
        self.mask_prob = 0.0
        self.no_mask_prob = 0.0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5
        )

        for (x, y, w, h) in faces:
            face = img[y:y + h, x:x + w]

            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face_resized = cv2.resize(face_rgb, (224, 224))
            face_resized = face_resized.astype("float32") / 255.0
            face_resized = np.expand_dims(face_resized, axis=0)

            prediction = model.predict(face_resized, verbose=0)[0]

            mask_prob = prediction[0] * 100
            no_mask_prob = prediction[1] * 100

            self.mask_prob = mask_prob
            self.no_mask_prob = no_mask_prob

            if mask_prob > no_mask_prob:
                label = "MASK"
                confidence = mask_prob
                color = (0, 255, 0)
            else:
                label = "NO MASK"
                confidence = no_mask_prob
                color = (0, 0, 255)

            cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
            cv2.putText(
                img,
                f"{label} {confidence:.2f}%",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# -----------------------------
# Start Streamer
# -----------------------------
ctx = webrtc_streamer(
    key="mask-detection",
    video_processor_factory=MaskDetector,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
)

# -----------------------------
# Live result display
# -----------------------------
result_placeholder = st.empty()

if ctx.video_processor:
    status_placeholder.success("Webcam active - detecting...")
    mask_prob = ctx.video_processor.mask_prob
    no_mask_prob = ctx.video_processor.no_mask_prob
    result_placeholder.markdown(
        f"### Mask: {mask_prob:.2f}%   |   No Mask: {no_mask_prob:.2f}%"
    )
else:
    status_placeholder.info("Click 'START' above to begin detection.")
    result_placeholder.markdown("### Waiting...")
