# Hand Gesture Recognition

Recognises hand gestures in real time from a webcam, for touch-free human–computer interaction.

## Two approaches

**1. MediaPipe gesture recognizer — `app2.py` (ready to run)**
Uses Google's pre-trained MediaPipe model (`gesture_recognizer.task`): it detects 21 hand landmarks per hand and classifies 7 gestures — Closed_Fist, Open_Palm, Pointing_Up, Thumb_Down, Thumb_Up, Victory, ILoveYou.

```bash
python app2.py                  # webcam, press q to quit
python app2.py --image hand.jpg # single image
```

**2. Custom CNN — `gesture_cnn_training.ipynb` + `app.py` (experiment)**
Gesture images were collected from a webcam, preprocessed (grayscale, background subtraction, 128×128) and used to train a Keras CNN. It reached very high validation accuracy, but because consecutive webcam frames are near-duplicates, frames of the same session appear in both train and validation sets, so that score is optimistic. A fair evaluation would split by recording session. `app.py` runs this model once `gesture_recognition_model.h5` has been trained.

## Tech Stack

Python · OpenCV · MediaPipe Tasks · TensorFlow / Keras

> **Credits:** the training notebooks come from a public GitHub repository by another author. This repository adds bug fixes, a working MediaPipe webcam app and this documentation.
