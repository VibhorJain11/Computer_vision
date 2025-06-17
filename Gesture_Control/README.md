# Hand Detection and Finger Tracking using MediaPipe & OpenCV

This project uses **MediaPipe** and **OpenCV** to detect hands in real-time using a webcam, and then:
- Identifies hand landmarks
- Determines how many fingers are up
- Calculates the distance between the thumb and index finger
- Dynamically visualizes this distance on-screen with a scaling bar

---

## Features
- Real-time hand tracking with up to 2 hands
- Bounding box and center of the hand displayed
- Finger state detection (up/down) using landmarks
- Measures distance between fingertips
- Scaled visual bar to represent finger distance
- Modular `HandDetector` class for easy reuse

---

## How It Works
- The hand landmarks are detected using **MediaPipe's `Hands` module**.
- Coordinates are scaled to the image resolution to draw landmarks and bounding boxes.
- A utility function counts fingers by comparing landmark positions.
- A visual **power bar** is drawn based on the distance between the thumb tip and index fingertip.

---
