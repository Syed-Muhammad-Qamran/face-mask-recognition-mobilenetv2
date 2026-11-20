# 😷 Face Mask Recognition using MobileNetV2 & Streamlit

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=Syed-Muhammad-Qamran/face-mask-recognition-mobilenetv2&branch=main&mainModule=app.py)
[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An automated **Face Mask Recognition System** powered by **MobileNetV2 Transfer Learning** and **OpenCV**, achieving approximately **98.7% validation accuracy**. 

Includes a cloud-ready **Streamlit web application** supporting:
* **📸 Snapshot Webcam Capture** (`st.camera_input`) - instant, works on all mobile & desktop browsers.
* **📁 Image Upload** (`st.file_uploader`) - analyze any photo from disk.
* **🎥 Real-Time WebRTC Stream** (`streamlit-webrtc`) - continuous live detection with bounding boxes.

---

## 🚀 1-Click Streamlit Cloud Deployment

You can deploy this application directly to **Streamlit Community Cloud** with one click:

[![Deploy to Streamlit Cloud](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=Syed-Muhammad-Qamran/face-mask-recognition-mobilenetv2&branch=main&mainModule=app.py)

### Manual Streamlit Cloud Setup Steps:
1. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
2. Click **"New app"**.
3. Select this repository: `Syed-Muhammad-Qamran/face-mask-recognition-mobilenetv2`.
4. Set **Branch:** `main`.
5. Set **Main file path:** `app.py`.
6. Click **"Deploy!"**.

Streamlit Cloud will read `.python-version` (Python 3.11) and `requirements.txt`, install dependencies, and launch your live web app in 2–3 minutes.

---

## 🧠 Model Architecture & Pipeline

```
Input Image / Video Frame
       │
       ▼
[OpenCV Haar Cascade Face Detector]
       │
       ▼
Crop & Preprocess Face (Resize 224x224, Normalize [0, 1])
       │
       ▼
[MobileNetV2 Base (Pre-trained on ImageNet)]
       │
       ▼
AveragePooling2D (7x7) -> Flatten -> Dense(128, ReLU) -> Dropout(0.5)
       │
       ▼
Dense(2, Softmax) Output
       │
       ├── Mask (Green Bounding Box + Confidence %)
       └── No Mask (Red Bounding Box + Confidence %)
```

* **Validation Accuracy:** $\approx 98.7\%$
* **Inference Speed:** Lightweight architecture optimized for mobile and real-time edge devices.

---

## 💻 Local Installation & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/Syed-Muhammad-Qamran/face-mask-recognition-mobilenetv2.git
cd face-mask-recognition-mobilenetv2
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python -m venv .venv

# Activate environment:
# Windows:
.venv\Scripts\activate
# Linux / macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Run the Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📁 Repository Structure

```text
face-mask-recognition-mobilenetv2/
│
├── app.py                      # Main Streamlit Cloud entrypoint (Webcam, Upload, WebRTC)
├── requirements.txt            # Streamlit Cloud headless dependencies
├── .python-version             # Pinned to Python 3.11 for TensorFlow compatibility
├── README.md                   # Project documentation & 1-click deploy badge
│
└── face_mask_detection/        # Original project directory
    ├── mask_detector.h5        # Pre-trained MobileNetV2 weights (11.5 MB)
    ├── accuracy_plot.png       # Training & validation performance curves
    ├── app.py                  # WebRTC-only app variant
    └── face_mask.py            # Original desktop Tkinter application
```

---

## 👤 Author
* **Syed Muhammad Qamran**
* GitHub: [@Syed-Muhammad-Qamran](https://github.com/Syed-Muhammad-Qamran)
