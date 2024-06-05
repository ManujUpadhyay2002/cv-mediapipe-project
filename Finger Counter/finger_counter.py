import cv2 as cv
import numpy as np
import mediapipe as mp
import imutils
import hand_tracking_module as htm

# Initialize the hand detector
detector = htm.handDetector()

# Capture video from the local camera (0 is usually the default camera)
cap = cv.VideoCapture(0)

# While loop to continuously fetch data from the local camera
while True:
    success, img = cap.read()
    if not success:
        break
    
    img = imutils.resize(img, width=1500, height=1900)
    img = detector.findHands(img=img)
    lm_dict = detector.findPosition(img=img, draw=False)
    
    if len(lm_dict) != 0:
        counter = 0
        if lm_dict[0]['x'] > lm_dict[4]['x']:
            cv.putText(img, 'Left Hand', (lm_dict[0]['x'] + 20, lm_dict[0]['y'] + 20), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
            # This For Left Hand
            if lm_dict[4]['x'] < lm_dict[3]['x']:
                counter += 1
                cv.circle(img, (lm_dict[4]['x'], lm_dict[4]['y']), 10, (255, 0, 255), -1)
        else:
            cv.putText(img, 'Right Hand', (lm_dict[0]['x'] + 20, lm_dict[0]['y'] + 20), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
            # This For Right Hand
            if lm_dict[4]['x'] > lm_dict[3]['x']:
                counter += 1
                cv.circle(img, (lm_dict[4]['x'], lm_dict[4]['y']), 10, (255, 0, 255), -1)
        if lm_dict[8]['y'] < lm_dict[6]['y']:
            counter += 1
            cv.circle(img, (lm_dict[8]['x'], lm_dict[8]['y']), 10, (255, 0, 255), -1)
        if lm_dict[12]['y'] < lm_dict[10]['y']:
            counter += 1
            cv.circle(img, (lm_dict[12]['x'], lm_dict[12]['y']), 10, (255, 0, 255), -1)
        if lm_dict[16]['y'] < lm_dict[14]['y']:
            counter += 1
            cv.circle(img, (lm_dict[16]['x'], lm_dict[16]['y']), 10, (255, 0, 255), -1)
        if lm_dict[20]['y'] < lm_dict[19]['y']:
            counter += 1
            cv.circle(img, (lm_dict[20]['x'], lm_dict[20]['y']), 10, (255, 0, 255), -1)
        cv.putText(img, f'{counter}', (100, 100), cv.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
    
    cv.imshow("Local Camera", img)

    # Press Esc key to exit
    if cv.waitKey(1) == 27:
        break

cap.release()
cv.destroyAllWindows()
