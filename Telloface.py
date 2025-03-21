import cv2
import numpy as np
from djitellopy import Tello
import time
from ultralytics import YOLO  # Assuming you use YOLOv5 or v8 for hand detection

class DroneController:
    def __init__(self):
        # Initialize the drone
        self.drone = Tello()
        self.drone.connect()
        print(f"Battery level: {self.drone.get_battery()}%")
        self.drone.streamon()
        self.drone.takeoff()
        time.sleep(2)  # Allow drone to stabilize

        # YOLO model for hand detection
        self.model = YOLO('/home/sacul/Tello/glovedetecion.pt')

        # PID control parameters and states
        self.frame_w, self.frame_h = 640, 480
        self.fbRange = [10000, 29000]  # Forward/Backward range
        self.pid_yaw = [0.3, 0.5, 0]
        self.pError = 0
        self.faceCascade = cv2.CascadeClassifier("/home/sacul/Tello/CBR-24/Tello/haarcascades/haarcascade_frontalface_default.xml")

    def find_face(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        myFaceListC, myFaceListArea = [], []
        for (x, y, w, h) in faces:
            cx, cy = x + w // 2, y + h // 2
            area = w * h
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
            myFaceListC.append([cx, cy])
            myFaceListArea.append(area)

        if myFaceListArea:
            i = myFaceListArea.index(max(myFaceListArea))
            return img, [myFaceListC[i], myFaceListArea[i]]
        else:
            return img, [[0, 0], 0]

    def track_face(self, info):
        area = info[1]
        x, y = info[0]
        fb = 0
        error = x - self.frame_w // 2
        speed = self.pid_yaw[0] * error + self.pid_yaw[1] * (error - self.pError)
        speed = int(np.clip(speed, -7, 7))

        # Adjust forward/backward velocity
        if self.fbRange[0] < area < self.fbRange[1]:
            fb = 0
        elif area > self.fbRange[1]:
            fb = -40  # Move backward faster if too close
        elif area < self.fbRange[0] and area != 0:
            fb = 40  # Move forward if too far

        # Reset if no face is detected
        if x == 0:
            speed = 0
            error = 0

        self.drone.send_rc_control(0, fb, 0, speed)
        return error

    def run(self):
        while True:
            img = self.drone.get_frame_read().frame
            img = cv2.resize(img, (self.frame_w, self.frame_h))
            img, info = self.find_face(img)
            self.pError = self.track_face(info)

            cv2.imshow("Drone Face Tracking", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.drone.land()
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    controller = DroneController()
    controller.run()
