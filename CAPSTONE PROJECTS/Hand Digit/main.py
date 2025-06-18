import numpy as np
import math
import cv2
import mediapipe as mp
from train import *

class HandDetector:
    """
    Finds Hands using the mediapipe library. Exports the landmarks
    in pixel format. Adds extra functionalities like finding how
    many fingers are up or the distance between two fingers. Also
    provides bounding box info of the hand found.
    """

    def __init__(self, staticMode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5):

        """
        :param mode: In static mode, detection is done on each image: slower
        :param maxHands: Maximum number of hands to detect
        :param modelComplexity: Complexity of the hand landmark model: 0 or 1.
        :param detectionCon: Minimum Detection Confidence Threshold
        :param minTrackCon: Minimum Tracking Confidence Threshold
        """
        self.staticMode = staticMode
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.detectionCon = detectionCon
        self.minTrackCon = minTrackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.staticMode,
                                        max_num_hands=self.maxHands,
                                        model_complexity=modelComplexity,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.minTrackCon)

        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.fingers = []
        self.lmList = []

    def findHands(self, img, draw=True, flipType=True):
        """
        Finds hands in a BGR image.
        :param img: Image to find the hands in.
        :param draw: Flag to draw the output on the image.
        :return: Image with or without drawings
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        allHands = []
        h, w, c = img.shape
        if self.results.multi_hand_landmarks:
            for handType, handLms in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
                myHand = {}
                ## lmList
                mylmList = []
                xList = []
                yList = []
                for id, lm in enumerate(handLms.landmark):
                    px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                    mylmList.append([px, py, pz])
                    xList.append(px)
                    yList.append(py)

                ## bbox
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                boxW, boxH = xmax - xmin, ymax - ymin
                bbox = xmin, ymin, boxW, boxH
                cx, cy = bbox[0] + (bbox[2] // 2), \
                         bbox[1] + (bbox[3] // 2)

                myHand["lmList"] = mylmList
                myHand["bbox"] = bbox
                myHand["center"] = (cx, cy)

                if flipType:
                    if handType.classification[0].label == "Right":
                        myHand["type"] = "Left"
                    else:
                        myHand["type"] = "Right"
                else:
                    myHand["type"] = handType.classification[0].label
                allHands.append(myHand)

                ## draw
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
                    cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),(bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),(255, 0, 255), 2)
                    cv2.putText(img, myHand["type"], (bbox[0] - 30, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN,2, (255, 0, 255), 2)

        return allHands, img
    def fingersUp(self, myHand):
        """
        Finds how many fingers are open and returns in a list.
        Considers left and right hands separately
        :return: List of which fingers are up
        """
        fingers = []
        myHandType = myHand["type"]
        myLmList = myHand["lmList"]
        if self.results.multi_hand_landmarks:

           
          
            if myLmList[self.tipIds[0]][0] > myLmList[self.tipIds[0] - 1][0]:
                fingers.append(1)
            else:
                fingers.append(0)
            
            # 4 Fingers
            for id in range(1, 5):
                if myLmList[self.tipIds[id]][1] < myLmList[self.tipIds[id] - 2][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        return fingers

    def findDistance(self, p1, p2, img=None, color=(255, 0, 255), scale=5):
        """
        Find the distance between two landmarks input should be (x1,y1) (x2,y2)
        :param p1: Point1 (x1,y1)
        :param p2: Point2 (x2,y2)
        :param img: Image to draw output on. If no image input output img is None
        :return: Distance between the points
                 Image with output drawn
                 Line information
        """

        x1, y1 = p1
        x2, y2 = p2
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)
        info = (x1, y1, x2, y2, cx, cy)
        if img is not None:
            cv2.circle(img, (x1, y1), scale, color, cv2.FILLED)
            cv2.circle(img, (x2, y2), scale, color, cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), color, max(1, scale // 3))
            cv2.circle(img, (cx, cy), scale, color, cv2.FILLED)

        return length, info, img


def main():
 
    cap = cv2.VideoCapture(0)

    # Initialize the HandDetector class with the given parameters
    detector = HandDetector(staticMode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5)


    img_canvas=np.zeros((480,640,3),np.uint8)
    

    img_canvas=cv2.flip(img_canvas,1)
    cv2.rectangle(img_canvas,(80,40),(140,100),(0,255,0),-1)
    cv2.rectangle(img_canvas,(80,140),(140,200),(255,0,0),-1)
    cv2.rectangle(img_canvas,(80,240),(140,300),(0,0,255),-1)
    cv2.rectangle(img_canvas,(80,340),(140,400),(255,255,255),-1)

    cv2.rectangle(img_canvas,(240,30),(630,440),(255,255,255),3)
    color=(255,255,255)
    thickness=15
    # Continuously get frames from the webcam
    while True:
        # Capture each frame from the webcam
        # 'success' will be True if the frame is successfully captured, 'img' will contain the frame
        success, img = cap.read()
        cv2.rectangle(img_canvas,(80,40),(140,100),(0,255,0),-1)
        cv2.rectangle(img_canvas,(80,140),(140,200),(255,0,0),-1)
        cv2.rectangle(img_canvas,(80,240),(140,300),(0,0,255),-1)
        cv2.rectangle(img_canvas,(80,340),(140,400),(255,255,255),-1)

        cv2.rectangle(img_canvas,(240,30),(630,440),(255,255,255),3)
       
        flag=False
        if not flag:
            img=cv2.flip(img,1)
            flag=True
        # Find hands in the current frame
        # The 'draw' parameter draws landmarks and hand outlines on the image if set to True
        # The 'flipType' parameter flips the image, making it easier for some detections
        hands, img = detector.findHands(img, draw=True, flipType=True)

        # Check if any hands are detected
        if hands:
            # Information for the first hand detected
            hand1 = hands[0]  # Get the first hand detected
            lmList1 = hand1["lmList"]  # List of 21 landmarks for the first handqqqq
            bbox1 = hand1["bbox"]  # Bounding box around the first hand (x,y,w,h coordinates)
            center1 = hand1['center']  # Center coordinates of the first hand
            handType1 = hand1["type"]  # Type of the first hand ("Left" or "Right")

            # Count the number of fingers up for the first hand
            fingers1 = detector.fingersUp(hand1)
            print(fingers1)  
            xp,yp=0,0
            x1,y1=lmList1[8][0],lmList1[8][1]
            print(x1,y1)
            if fingers1[1]==1 and fingers1[2]==0:
                if xp==0 and yp==0:
                    xp,yp=x1,y1
                # cv2.line(img,(xp,yp),(lmList1[8][1],lmList1[8][2]),(0,255,0),2)
                cv2.line(img_canvas,(xp,yp),(x1,y1),color,thickness)
                (xp,yp)=x1,y1

            
            if fingers1[1]==1 and fingers1[2]==1:
                if x1>80 and x1<140:
                    if y1>40 and y1<100:
                        print('green selected')
                        color=(0,255,0)
                        thickness=20
                
                    if y1>140 and y1<200:
                        print('blue selected')
                        color=(255,0,0)
                        thickness=20
                
                    if y1>240 and y1<300:
                        print('red selected')
                        color=(0,0,255)
                        thickness=20
                
                    if y1>340 and y1<400:
                        print('eraser selected')
                        color=(0,0,0)
                        thickness=50
                
            print(" ") 

        from tensorflow.keras.models import load_model
        model = load_model("cnn_digit_model.h5")

        img_G = cv2.cvtColor(img_canvas, cv2.COLOR_BGR2GRAY)
        img_G = cv2.GaussianBlur(img_G, (5, 5), 0)
        _, img_inv = cv2.threshold(img_G, 100, 255, cv2.THRESH_BINARY_INV)  # Increase threshold to remove background noise

        img_inv=cv2.cvtColor(img_inv,cv2.COLOR_GRAY2BGR)
        
        img=cv2.bitwise_and(img,img_inv)
        img=cv2.bitwise_or(img,img_canvas)

        roi = img_G[45:435, 235:635]
        roi=cv2.resize(roi,(28,28))
        roi=255-roi
       
        # Display the image in a window
        cv2.imshow("Image", img)
        cv2.imshow("Canvas", img_canvas)
        cv2.imshow("roi", roi)

        input_image = roi
        input_image_resize = input_image/255.0

        cv2.imshow("final_image", input_image_resize)

        image_reshaped = np.reshape(input_image_resize, [1,28,28,1])
        input_prediction = model.predict(image_reshaped)
        input_pred_label = np.argmax(input_prediction)

        print('The Handwritten Digit is recognised as ', input_pred_label)


        # Keep the window open and update it for each frame; wait for 1 millisecond between frames
         
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break


if __name__ == "__main__":
    main()
