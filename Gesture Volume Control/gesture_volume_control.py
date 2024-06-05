import cv2 as cv
import numpy as np
import mediapipe as mp
import imutils
import time
import hand_tracking_module as htm
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

def convert_to_dB(computer_volume):
    if computer_volume <= 0:
        dB_volume = -float('inf')
    elif computer_volume > 100:
        dB_volume = 0.0
    else:
        dB_volume = (computer_volume - 100) / 2.6667
    return dB_volume

# Initialize the hand detector
detector = htm.handDetector()

# Set up audio utilities
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]
volBar = 400
volPer = 0

# Initialize webcam
cap = cv.VideoCapture(0)  # 0 is usually the default camera

while True:
    success, img = cap.read()
    if not success:
        break
    
    img = imutils.resize(img, width=1000, height=1800)
    img = detector.findHands(img=img)
    lm_dict = detector.findPosition(img=img, draw=False)
    
    if len(lm_dict) != 0:
        pt1 = np.array([lm_dict[4]['x'], lm_dict[4]['y']])
        pt2 = np.array([lm_dict[8]['x'], lm_dict[8]['y']])
        dif = np.linalg.norm(pt2 - pt1)
        cv.line(img, pt1=pt1, pt2=pt2, color=(255, 0, 255), thickness=2)
        cv.circle(img, pt1, 10, (255, 0, 255), -1)
        cv.circle(img, pt2, 10, (255, 0, 255), -1)
        cv.circle(img, ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2), 10, (255, 255, 255), -1)
        computerVol = 0.40 * dif
        volBar = np.interp(dif, [30, 260], [400, 150])
        volPer = np.interp(dif, [30, 260], [0, 100])
        dB_volume = convert_to_dB(computerVol)
        volume.SetMasterVolumeLevel(dB_volume, None)
    
    cv.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
    cv.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), -1)
    cv.putText(img, f'{int(volPer)}%', (45, 500), cv.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (255), 1)
    cv.imshow("Webcam", img)

    # Press Esc key to exit
    if cv.waitKey(1) == 27:
        break

cap.release()
cv.destroyAllWindows()
