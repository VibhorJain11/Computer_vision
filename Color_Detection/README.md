# Real-Time Color Detection Using OpenCV 

This project uses OpenCV and NumPy to detect a specific color (in this case, yellow) in real-time from your webcam feed. The code identifies the target color by converting the frame from BGR to HSV color space and applying a mask to highlight only the desired color region.

##  Features
- Real-time video processing using webcam
- HSV-based color filtering
- Bitwise masking to extract only the desired color
- Easy to modify for detecting other colors

##  How It Works
1. The color to be detected is defined in BGR format.
2. It's converted to HSV and a range is calculated around its hue.
3. Every frame from the webcam is converted to HSV.
4. A mask is applied to highlight areas matching the color range.
5. The result is displayed in real-time.

##  Detected Color
- **Yellow**
- BGR: `[0, 140, 255]`
- You can modify this color to detect red, green, blue, etc.
