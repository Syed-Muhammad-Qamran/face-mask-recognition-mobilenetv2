import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model

# Load Trained Model
model = load_model("mask_detector.h5")

# Classes
class_names = ["Mask", "No Mask"]

# Face Detector
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# GUI Window
root = tk.Tk()
root.title("Face Mask Detection System")
root.geometry("900x700")

title = tk.Label(
    root,
    text="Face Mask Detection",
    font=("Arial", 20, "bold")
)
title.pack(pady=10)

video_label = tk.Label(root)
video_label.pack()

result_label = tk.Label(
    root,
    text="Waiting...",
    font=("Arial", 14, "bold")
)
result_label.pack(pady=10)

# Webcam
cap = cv2.VideoCapture(0)


def update_frame():
    ret, frame = cap.read()

    if ret:

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5
        )

        for (x, y, w, h) in faces:

            face = frame[y:y+h, x:x+w]

            face_rgb = cv2.cvtColor(
                face,
                cv2.COLOR_BGR2RGB
            )

            face_resized = cv2.resize(
                face_rgb,
                (224, 224)
            )

            face_resized = face_resized.astype("float32") / 255.0

            face_resized = np.expand_dims(
                face_resized,
                axis=0
            )

            prediction = model.predict(
                face_resized,
                verbose=0
            )[0]

            mask_prob = prediction[0] * 100
            no_mask_prob = prediction[1] * 100

            if mask_prob > no_mask_prob:
                label = "MASK"
                confidence = mask_prob
                color = (0, 255, 0)
            else:
                label = "NO MASK"
                confidence = no_mask_prob
                color = (0, 0, 255)

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                color,
                3
            )

            cv2.putText(
                frame,
                f"{label} {confidence:.2f}%",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

            result_label.config(
                text=f"Mask: {mask_prob:.2f}%   |   No Mask: {no_mask_prob:.2f}%"
            )

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        img = Image.fromarray(frame_rgb)

        img = img.resize((800, 500))

        imgtk = ImageTk.PhotoImage(image=img)

        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

    root.after(10, update_frame)


def close_window():
    cap.release()
    root.destroy()


exit_button = tk.Button(
    root,
    text="Exit",
    font=("Arial", 12, "bold"),
    bg="red",
    fg="white",
    command=close_window
)

exit_button.pack(pady=10)

root.protocol("WM_DELETE_WINDOW", close_window)

update_frame()

root.mainloop()
