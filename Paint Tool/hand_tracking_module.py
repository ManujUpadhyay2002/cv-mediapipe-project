import cv2 as cv
import numpy as np
import requests
import imutils
import time
import mediapipe as mp


class handDetector():
    def __init__(self, mode=False, maxHands=2, modelC=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelC = modelC
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        # This is For detecting Hand
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            self.mode, self.maxHands, self.modelC, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # This only work on RGB
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        self.lmDict = {}
        if self.results.multi_hand_landmarks:
            # This is for one hand
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                self.lmDict[id] = {'x': cx, 'y': cy}
                if draw:
                    cv.circle(img, (cx, cy), 5, (255, 0, 255), -1)
        return self.lmDict

    def fingerCount(self):
        counter = [0,0,0,0,0]
        if self.lmDict[0]['x'] > self.lmDict[4]['x']:
            if self.lmDict[4]['x'] < self.lmDict[3]['x']:
                counter[0] = 1
        else:
            if self.lmDict[4]['x'] > self.lmDict[3]['x']:
                counter[0] = 1
        if self.lmDict[8]['y'] < self.lmDict[6]['y']:
            counter[1] = 1
        if self.lmDict[12]['y'] < self.lmDict[10]['y']:
            counter[2] = 1
        if self.lmDict[16]['y'] < self.lmDict[14]['y']:
            counter[3] = 1
        if self.lmDict[20]['y'] < self.lmDict[19]['y']:
            counter[4] = 1
        return counter

def main():
    pass

    # Test Cod:
    # pTime = 0
    # cTime = 0
    # url = 'http://100.70.46.95:8080//shot.jpg'
    # detector = handDetector()
    # while True:
    #     img_resp = requests.get(url)
    #     img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    #     img = cv.imdecode(img_arr, -1)
    #     img = imutils.resize(img, width=1000, height=1800)
    #     img = detector.findHands(img=img)
    #     lmList = detector.findPosition(img)
    #     if len(lmList) !=0:
    #         print(lmList[4])
    #     cTime = time.time()
    #     fps = 1/(cTime-pTime)
    #     pTime = cTime
    #     cv.putText(img, str(int(fps)), (10, 70),
    #                cv.FONT_HERSHEY_COMPLEX, 3, (255, 0, 255), 2)
    #     cv.imshow("Android_cam", img)
    #     if cv.waitKey(1) == 27:
    #         break


if __name__ == '__main__':
    main()
