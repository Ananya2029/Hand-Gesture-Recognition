"""Webcam demo for the custom CNN trained in Untitled.ipynb.

Needs the trained model file `gesture_recognition_model.h5` in this folder
(produced by the notebook; not included because of its size).
For a ready-to-run demo, use app2.py (MediaPipe) instead.
"""
from pathlib import Path

import cv2
import numpy as np
from tensorflow.keras.models import load_model

MODEL_PATH = Path(__file__).resolve().parent / "gesture_recognition_model.h5"
if not MODEL_PATH.exists():
    raise SystemExit(
        f"Model not found at {MODEL_PATH}.\n"
        "Train it with Untitled.ipynb, or run the MediaPipe version: python app2.py"
    )

model = load_model(MODEL_PATH)

# Define the size to which images will be resized
resize_width = 128
resize_height = 128

# List of gestures
gestures = [
    "like", "dislike", "peace", "one", "fist", "Hello", "Love you"
]

# Create the background subtractor once so it learns the background across frames
back_sub = cv2.createBackgroundSubtractorMOG2()


def preprocess_image(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fg_mask = back_sub.apply(gray)                       # background subtraction
    fg_image = cv2.bitwise_and(frame, frame, mask=fg_mask)
    resized_img = cv2.resize(fg_image, (resize_width, resize_height))
    resized_img = resized_img.astype("float32") / 255.0
    return np.expand_dims(resized_img, axis=0)


def predict_gesture(frame):
    prediction = model.predict(preprocess_image(frame), verbose=0)
    return gestures[int(np.argmax(prediction))]


# Start the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gesture = predict_gesture(frame)

    # Draw a rectangle around the largest contour (assumed to be the hand)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, gesture, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Hand Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
