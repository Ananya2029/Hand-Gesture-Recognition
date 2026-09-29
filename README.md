# Hand Gesture Recognition

> Task 4 of my **Machine Learning internship at [Prodigy InfoTech](https://prodigyinfotech.dev/)**.

Recognises hand gestures in real time from a webcam, for touch-free human–computer interaction.

## Two approaches

**1. MediaPipe gesture recognizer — `app2.py` (ready to run)**
Uses Google's pre-trained MediaPipe model (`gesture_recognizer.task`): it detects 21 hand landmarks per hand and classifies 7 gestures — Closed_Fist, Open_Palm, Pointing_Up, Thumb_Down, Thumb_Up, Victory, ILoveYou.

```bash
python app2.py                  # webcam, press q to quit
python app2.py --image hand.jpg # single image
```

**2. Custom CNN — `Untitled.ipynb` + `app.py` (experiment)**
I collected my own gesture images from the webcam, preprocessed them (grayscale, background subtraction, 128×128) and trained a Keras CNN. It reached very high validation accuracy, but because consecutive webcam frames are near-duplicates, frames of the same session appear in both train and validation sets, so that score is optimistic. A fair evaluation would split by recording session. `app.py` runs this model once `gesture_recognition_model.h5` has been trained.

## Tech Stack

Python · OpenCV · MediaPipe Tasks · TensorFlow / Keras
