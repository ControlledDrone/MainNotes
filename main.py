# Import the necessary modules
from djitellopy import tello
import cv2
import time
import numpy as np
import math
import mediapipe as mp

# Connect to our drone
me = tello.Tello()
me.connect()


class HandDetector():
    # Constructor for our hand detector
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    # Method finds hands in img and return img with hands drawn
    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
        return img

    # Method used to return list of landmarks in img
    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 2, (255, 0, 255), cv2.FILLED)
        return lmList


# Class used to make calculations to control the drone and get info on the drone
class DroneControls():
    # Method used to find the value for how much battery is left on the drone
    def tello_battery(tello):
        global battery_status
        battery_status = me.get_battery()
        return int(battery_status)

    # Method creates deadzone in the center of the img when a certain landmark is inside
    def createDeadZone(self, img, lmPos):
        x, y = lmPos[0:]
        h, w = img.shape[:2]
        screen_center_w = w / 2
        screen_center_h = h / 2

        dead_zone_size = 100
        out_of_dz = True

        # If landmark is inside deadzone return out_of_dz==false
        if x < screen_center_w + dead_zone_size and x > screen_center_w - dead_zone_size and y < screen_center_h + dead_zone_size and y > screen_center_h + dead_zone_size:
            return out_of_dz == False

    # Method used to find the velocity from the center of the img to a certain landmark
    def findCenterVelo(self, img, lmPos):
        x, y = lmPos[0:]
        h, w = img.shape[:2]
        center_dist = np.linalg.norm(np.array((x, y)) - np.array((w / 2, h / 2)))
        velocity = (center_dist / w) * 3.0
        # Draws velocity line in opencv
        cv2.line(img, (x, y), (int(w / 2), int(h / 2)), (255, 0, 0), 5)
        return int(velocity)

    # Method used to find the distance between two landmarks
    def findDistanceLms(self, lmPos1, lmPos2):
        x1, y1 = lmPos1[1:]
        x2, y2 = lmPos2[1:]
        # Calculates distance - https://morioh.com/p/9ce670a59fc3
        distance = math.hypot(x2 - x1, y2 - y1)
        return distance

    # Method used to find the middle coordinate between two landmarks
    def findMiddleCoordinate(self, lmPos1, lmPos2):
        x1, y1 = lmPos1[1:]
        x2, y2 = lmPos2[1:]
        # Calculates the middle coordinate between the two landmarks
        x, y = (x1 + x2) // 2, (y1 + y2) // 2
        return x, y

    # Method used to return the x and y values for a certain landmark
    def findXY(self, lmPos1):
        x1, y1 = lmPos1[1:]
        return x1, y1

    # Method used to find open fingers 
    def findNoOfFingers(self, img, lmList, tipIds):
        fingers = []

        if len(lmList) !=0  in img:
            if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)

            # Loop for fingers minus thumb
            for id in range(0, 5):
                if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            noOfFingers = fingers.count(1)
            print(noOfFingers)

        return noOfFingers


# Class used to make the drone fly in certain directions
class NavigateDrone():
    # Method used for navigating the drone 
    def navigateDrone(self, noOfFingers, velocity, distance1):
        lr, fb, ud, yv = 0, 0, 0, 0 #LeftRight, ForwardBackward, UpDown, Yaw (side to side)
        speed = 10 

        # If zero fingers are up 
        if noOfFingers == 0:
            print("Landing the Drone... Stand Clear")
            me.land()

        # if thumb, pointer finger and middle finger
        if noOfFingers == 3:
            print("Taking off... Stand Clear")
            me.takeoff()

        # If thumb
        if noOfFingers == 1:
            print("Flying down...")
            ud = -speed-(velocity*10)

        # if thumb and pointer finger 
        if noOfFingers == 2:
            print("Flying up...")
            ud = +speed+(velocity*10)
            
        # Forwards
        if noOfFingers == 5 and distance1 < 140 and distance1 > 110:
            fb = -speed-10
            print("Flying forwards x 10 Speed")
        elif distance1 < 110 and distance1 > 80:
            fb = -speed-20
            print("Flying forwards x 20 Speed")
        elif distance1 < 80 and distance1 > 50 :
            fb = -speed-30
            print("Flying forwards x 30 Speed")
        
        # Backwards
        if noOfFingers == 5 and distance1 > 160 and distance1 < 190:
            fb = speed+10
            print("Flying backwards x 10 Speed")
        elif distance1 > 190 and distance1 < 220:
            fb = speed+20
            print("Flying backwards x 20 Speed")
        elif distance1 > 220 and distance1 < 250 :
            fb = speed+30
            print("Flying backwards x 30 Speed")

        vals = lr, fb, ud, yv

        # To be able to senc rc controls to tello drone
        return me.send_rc_control(vals[0], vals[1], vals[2], vals[3])


# Class used to run our graphical user interface with opencv
class GUI():
    def startGUI():
        # Class objects
        detector = HandDetector()
        drone = DroneControls()
        navDrone = NavigateDrone()

        # Variables used for calculating fps (frames per second)
        pTime = 0
        cTime = 0

        # Capture video from webcam
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        # Set width and height
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        battery_status = drone.tello_battery()

        while True:
            success, img = cap.read()
            # Flips img horizontally to create a mirror effect
            img = cv2.flip(img, 1)
            # Finds hands in the webcamera img
            img = detector.findHands(img)
            # Finds landmarks in the webcamera img
            lmList = detector.findPosition(img)
            tipIds = [4, 8, 12, 16, 20]

            if len(lmList) != 0:
                # Finds coordinate for center of hand
                centerCoordinate = drone.findMiddleCoordinate(lmList[9], lmList[0])

                # Draws and finds line for velocity
                velocity = drone.findCenterVelo(img, centerCoordinate)

                # Creates deadzone that activates when centercoordinate is inside
                drone.createDeadZone(img, centerCoordinate)

                # Draws line used for depth
                x9, y9 = drone.findXY(lmList[9])
                x0, y0 = drone.findXY(lmList[0])
                cv2.line(img, (x9, y9), (x0, y0), (255, 0, 255), 3)
                # Finds distance between two landmarks
                distance1 = drone.findDistanceLms(lmList[9], lmList[0])

                #Finds number of fingers 
                noOfFingers = drone.findNoOfFingers(img, lmList, tipIds)

                # Navigate the drone forwards and backwards witg the depth line
                navDrone.navigateDrone(noOfFingers, velocity, distance1)

            # Calculates fps (frames per second)
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime

            # Add text with fps to opencv
            cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                        (255, 0, 255), 3)

            # Add text with battery power to opencv
            cv2.putText(img, "Battery: {}".format(battery_status) + "%", (5, 435),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("Image", img)

            k = cv2.waitKey(1)
            if k & 0xFF == ord('q'):  # close on key 'q'
            #if cv2.waitKey(1) % 256 == 27:  # Esc key       # Kinda stupid slow to close down

                print("Closing")
                break

        # Release webcamera and close all opencv windows
        cap.release()
        cv2.destroyAllWindows()


# The main function
def main():
    print("Starting GUI...")
    # Begin method startGUI() from class GUI()
    GUI.startGUI()


if __name__ == "__main__":
    main()
