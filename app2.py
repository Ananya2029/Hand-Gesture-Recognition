"""Real-time hand gesture recognition with MediaPipe Tasks.

Uses Google's pre-trained gesture recognizer (gesture_recognizer.task), which detects
21 hand landmarks and classifies 7 gestures: Closed_Fist, Open_Palm, Pointing_Up,
Thumb_Down, Thumb_Up, Victory, ILoveYou.

Usage:
    python app2.py                 # webcam (press q to quit)
    python app2.py --image hand.jpg
"""
import argparse
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = Path(__file__).resolve().parent / "gesture_recognizer.task"

# Landmark index pairs that form the hand skeleton (thumb, fingers, palm)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]


def create_recognizer(running_mode):
    options = vision.GestureRecognizerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=running_mode,
        num_hands=2,
        min_hand_detection_confidence=0.6,
    )
    return vision.GestureRecognizer.create_from_options(options)


def draw_results(frame, result):
    """Draw landmarks, skeleton and gesture label for every detected hand."""
    h, w = frame.shape[:2]
    for i, landmarks in enumerate(result.hand_landmarks):
        points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, points[a], points[b], (255, 255, 255), 2)
        for p in points:
            cv2.circle(frame, p, 4, (0, 200, 0), -1)

        if result.gestures and result.gestures[i]:
            top = result.gestures[i][0]
            label = f"{top.category_name} ({top.score:.0%})"
            x, y = min(p[0] for p in points), min(p[1] for p in points)
            cv2.putText(frame, label, (x, max(y - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return frame


def run_on_image(path):
    frame = cv2.imread(path)
    if frame is None:
        raise SystemExit(f"Could not read image: {path}")
    with create_recognizer(vision.RunningMode.IMAGE) as recognizer:
        image = mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = recognizer.recognize(image)
    gestures = [g[0].category_name for g in result.gestures if g]
    print("Detected gestures:", gestures or "no hand found")
    cv2.imshow("Hand Gesture Recognition", draw_results(frame, result))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_on_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("No webcam found.")
    start = time.monotonic()
    with create_recognizer(vision.RunningMode.VIDEO) as recognizer:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)  # mirror view feels natural
            image = mp.Image(image_format=mp.ImageFormat.SRGB,
                             data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            timestamp_ms = int((time.monotonic() - start) * 1000)
            result = recognizer.recognize_for_video(image, timestamp_ms)
            cv2.imshow("Hand Gesture Recognition (q to quit)", draw_results(frame, result))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="Run on an image file instead of the webcam")
    args = parser.parse_args()
    run_on_image(args.image) if args.image else run_on_webcam()
