import cv2 as cv
import numpy as np
import mediapipe as mp
import imutils
import hand_tracking_module as htm

# Initialize hand detector
detector = htm.handDetector()
paintImg = cv.imread('board.jpg')
drawColor, xp, yp = 0, 0, 0
imgCanvas = np.zeros((562, 1000, 3), np.uint8)

# Open the webcam
cap = cv.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    img = imutils.resize(img, width=1000, height=1800)
    img = cv.flip(img, 1)
    # Resize imgCanvas to match the dimensions of img
    imgCanvas = cv.resize(imgCanvas, (img.shape[1], img.shape[0]))
    img[:100, :1000] = paintImg[:100, :1000]
    img = detector.findHands(img=img)
    lm_dict = detector.findPosition(img=img, draw=False)

    if len(lm_dict) != 0:
        pt1 = np.array([lm_dict[8]['x'], lm_dict[8]['y']])
        pt2 = np.array([lm_dict[12]['x'], lm_dict[12]['y']])
        fingers = detector.fingerCount()
        
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            if 170 < pt2[0] < 360 and pt2[1] < 100:
                drawColor = (0, 0, 255)
            elif 370 < pt2[0] < 550 and pt2[1] < 100:
                drawColor = (255, 0, 255)
            elif 585 < pt2[0] < 780 and pt2[1] < 100:
                drawColor = (255, 0, 0)
            elif 785 < pt2[0] < 1000 and pt2[1] < 100:
                drawColor = (0, 0, 0)
            cv.rectangle(img, pt1, pt2, drawColor, 2)
        
        if fingers[1] and fingers[2] == False:
            cv.circle(img, pt1, 15, drawColor, -1)
            if xp == 0 and yp == 0:
                xp, yp = pt1
            if drawColor == (0, 0, 0):
                cv.line(img, (xp, yp), pt1, drawColor, 100)
                cv.line(imgCanvas, (xp, yp), pt1, drawColor, 100)
            else:
                cv.line(img, (xp, yp), pt1, drawColor, 15)
                cv.line(imgCanvas, (xp, yp), pt1, drawColor, 15)
            xp, yp = pt1

    imgGray = cv.cvtColor(imgCanvas, cv.COLOR_BGR2GRAY)
    _, imgInv = cv.threshold(imgGray, 50, 255, cv.THRESH_BINARY_INV)
    imgInv = cv.cvtColor(imgInv, cv.COLOR_GRAY2BGR)
    img = cv.bitwise_and(img, imgInv)
    img = cv.bitwise_or(img, imgCanvas)
    
    cv.imshow("Laptop Camera", img)

    # Press Esc key to exit
    if cv.waitKey(1) == 27:
        break

cap.release()
cv.destroyAllWindows()
