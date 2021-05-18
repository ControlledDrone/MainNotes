import cv2
import numpy as np
import time
from djitellopy import tello
import mediapipe as mp
import math

me = tello.Tello()
me.connect()

def tello_battery(tello): 
    global battery_status
    battery_status = me.get_battery()
    return int(battery_status)

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 2, (255, 0, 255), cv2.FILLED)
        return lmList


def control_drone(img, handPos):
    x, y = handPos[1:]
    h, w = img.shape[:2]
    screen_center_w = w / 2
    screen_center_h = h / 2

    # create and area where no message is sent
    dead_zone_size = 50
    if x < screen_center_w + dead_zone_size and x > screen_center_w - dead_zone_size \
            and y < screen_center_h + dead_zone_size and y > screen_center_h - dead_zone_size:
        # can is in center
        return

    # calculate distance from center
    # https://stackoverflow.com/a/1401828/4560132
    center_dist = np.linalg.norm(np.array((x, y)) - np.array((w / 2, h / 2)))
    velocity = (center_dist / w) * 3.0

    # draw distance to center
    # +50 to go 50 pixels down in the screen from landmark position

    cv2.line(img, (x, y), (int(w / 2), int(h / 2)), (255, 0, 0), 5)

    # create message
    #if x > screen_center_w:
        #print(f'RIGHT {velocity}')
        # me.takeoff()
    #elif x < screen_center_w:
        #print(f'LEFT {velocity}')
        # me.land()

def start_cv():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    detector = handDetector()
    battery_status = tello_battery(tello)
    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)  # mirror image
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            control_drone(img, lmList[9])

            x1, y1 = lmList[9][1], lmList[9][2]
            x2, y2 = lmList[0][1], lmList[0][2]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            cv2.circle(img, (x1, y1), 10, (0, 0, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (0, 0, 0), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            #cv2.circle(img, (cx, cy), 10, (0, 0, 0), cv2.FILLED)

            # https://morioh.com/p/9ce670a59fc3
            length = math.hypot(x2 - x1, y2 - y1)
            #print(length)

            if length < 130 :
                print("Flying backwards x 20 Speed")

        else:
            print("DRONE IS HOVERING")

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)

        cv2.putText(img, "Battery: {}".format(battery_status)+"%", (5, 435),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Image", img)

        k = cv2.waitKey(1)
        if k & 0xFF == ord('q'):  # close on key 'q'
            print("Closing")
            break
    me.end()
    cap.release()
    cv2.destroyAllWindows()


# The main function
def main():
    "Starting GUI..."
    start_cv()


if __name__ == "__main__":
    main()
