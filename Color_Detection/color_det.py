import numpy as np
import cv2


def get_limits(color):
    # Convert the BGR color to a 1x1 image
    bgr_color = np.uint8([[color]])

    # Convert BGR to HSV
    hsv_color = cv2.cvtColor(bgr_color, cv2.COLOR_BGR2HSV)

    # Extract the hue value
    hue = hsv_color[0][0][0]

    # Define HSV range
    lowerLimit = (hue - 10, 100, 100)
    upperLimit = (hue + 10, 255, 255)

    # Convert to NumPy arrays
    lowerLimit = np.array(lowerLimit, dtype=np.uint8)
    upperLimit = np.array(upperLimit, dtype=np.uint8)

    return lowerLimit, upperLimit

cap=cv2.VideoCapture(0)


    
while True:
    ret,frame=cap.read()
    
    

    
    yellow=[0,140,255]
    lowerl,upperl=get_limits(yellow)
    hsvI=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    masked=cv2.inRange(hsvI,lowerl,upperl)
    img=cv2.bitwise_and(frame,frame,mask=masked)
    cv2.imshow('masked',img)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
    # maskedI=Image.fromarray(masked)
    # box=maskedI.getbbox()
    # if box!=None:
        
    #     x1,y1,x2,y2=box
    #     cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            
cap.release()
cv2.destroyAllWindows()# This is the code for color detection module
