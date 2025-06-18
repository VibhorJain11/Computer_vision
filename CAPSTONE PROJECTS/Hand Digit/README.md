# Multi-Modal Hand Tracking and CNN-Based Digit Recognition Pipeline

## Overview
This project implements a real-time hand gesture-based digit recognition system using Mediapipe for hand tracking, OpenCV for image processing, and a CNN model for digit classification. The system detects hand landmarks, tracks finger positions for drawing on a virtual canvas, and recognizes handwritten digits from the canvas.

## Features
- Real-time hand detection and landmark extraction with Mediapipe.
- Finger state recognition to enable drawing and color selection.
- Virtual canvas drawing with selectable brush colors and eraser.
- Digit recognition using a trained CNN on drawn digits.
- ROI extraction and preprocessing for model input.

## Requirements
- Python 3.7+
- OpenCV
- Mediapipe
- TensorFlow
- NumPy

