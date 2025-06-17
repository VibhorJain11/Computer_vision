import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector

import numpy as np
import math

# Webcam setup
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Hand detector initialization
detector = HandDetector(detectionCon=0.8)
colorR = (255, 0, 255)

# Class for draggable rectangles
class DragRect:
    def __init__(self, posCenter, size=[200, 200]):
        self.posCenter = posCenter
        self.size = size
        self.lastlen = 0 

    def update(self, cursor):
        cx, cy = self.posCenter
        w, h = self.size

        # If the cursor (index fingertip) is inside the rectangle
        if cx - w // 2 < cursor[0] < cx + w // 2 and \
                cy - h // 2 < cursor[1] < cy + h // 2:
            self.posCenter = cursor

    def chngesize(self,dist,prevdist):
        # l, _, _ = detector.findDistance(lmList[8][:2], lmList[4][:2], img)
        
        w, h = self.size
        if dist-prevdist>0:

            w=w*(1+(dist-prevdist)/dist)
            h=h*(1+(dist-prevdist)/dist)
        else:
            w=w*(1-(prevdist-dist)/dist)
            h=h*(1-(prevdist-dist)/dist)

        self.size=int(w), int(h) 


# Create multiple rectangles
rectList = [DragRect([x * 250 + 150, 150]) for x in range(5)]

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)  # Flip for a mirror effect

    # Find hands and landmarks
    hands, img = detector.findHands(img)
    if hands:
        # Get landmarks for the first detected hand
        lmList = hands[0]["lmList"]

        # Calculate distance between index and middle finger tips
        length, _, _ = detector.findDistance(lmList[8][:2], lmList[4][:2], img)
        length2, _, _ = detector.findDistance(lmList[8][:2], lmList[12][:2], img)

        # If distance is small, consider it a "pinch" gesture
        if length < 30:
            cursor = lmList[8][:2]  # Get index finger tip position
            for rect in rectList:
                rect.update(cursor)
        
        if length2 < 30:
            cursor = lmList[8][:2] 
            
             # Get index finger tip position
            for rect in rectList:
                cx, cy = rect.posCenter
                w, h = rect.size
                if cx - w // 2 < cursor[0] < cx + w // 2 and \
                cy - h // 2 < cursor[1] < cy + h // 2:
                    if rect.lastlen==0:
                        rect.lastlen=length
                    else:
                        if abs(rect.lastlen-length)>25:
                            
                            rect.chngesize(length,rect.lastlen)
                            rect.lastlen=length
                        else:continue
        else:
            for rect in rectList:
                rect.lastlen=0
    def dist(point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    def sigmoid(x):
        
        return 1 / (1 + math.exp(-x))
    # Draw rectangles with transparency
    imgNew = np.zeros_like(img, np.uint8)
    for rect in rectList:

        cx, cy = rect.posCenter
        w, h = rect.size
        scaled_w = sigmoid((w - 100) / 22)  # Adjust scaling factor (100 and 50) based on your rectangle size range
        scaled_h = sigmoid((h - 100) / 22)

        # Map sigmoid output to color range (0-255)
        red = int(255 * scaled_w)  # Red intensity based on width
        blue = int(255 * scaled_h)  
        cv2.rectangle(imgNew, (cx - w // 2, cy - h // 2),
                      (cx + w // 2, cy + h // 2), (0, 0, blue), cv2.FILLED)
        cvzone.cornerRect(imgNew, (cx - w // 2, cy - h // 2, w, h), 20, rt=0)


    center = (200, 180)  # Coordinates of the circle's center (x, y)
    radius = 100         # Radius of the circle
    color = (255, 0, 0)  # Blue color in BGR format
    thickness = -1       # Fill the circle

    # Draw the circle directly on `img`
    cv2.circle(img, center, radius, color, thickness)
    for rect in rectList:
        cx, cy = rect.posCenter
        if center[0]-20<cx<center[0]+20 and center[1]-20<cy<center[1]+20:
            rectList.remove(rect)
    # Display result
    cv2.imshow("Image", img)
        
    # Add transparency overlay
    alpha = 0.5
    mask = imgNew.astype(bool)
    img[mask] = cv2.addWeighted(img, alpha, imgNew, 1 - alpha, 0)[mask]

    # Display result
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
