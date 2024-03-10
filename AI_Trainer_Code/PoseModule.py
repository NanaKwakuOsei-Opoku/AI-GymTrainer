import cv2
import mediapipe as mp
import time
import math
import numpy

class poseDetector():
    def __init__(self, mode=False, upBody=False, smooth=True ,detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.MyPose = self.mpPose.Pose(self.mode, self.upBody, self.smooth, self.detectionCon, self.trackCon)

    def drawPose(self, image, draw=True):
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.MyPose.process(imageRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(image, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return image


    def return_Positions(self, image):
        self.positions = []   # it is an array of Tupules
        if self.results.pose_landmarks:
            for id, landmarks in enumerate(self.results.pose_landmarks.landmark):
                height, width, c = image.shape
                x, y = int(landmarks.x * width), int(landmarks.y * height)
                self.positions.append([id, x, y])
                cv2.circle(image, (x, y), 5, (255, 0, 0), cv2.FILLED)
        return self.positions

    def findAngle(self, image, id1, id2, id3): # based on the index values we should retrieve the respective points
        if len(self.positions) > max(id1, id2, id3):
            x1, y1 = self.positions[id1][1:]
            x2, y2 = self.positions[id2][1:]
            x3, y3 = self.positions[id3][1:]

            cv2.circle(image, (x1, y1), 7, (0, 0, 255), cv2.FILLED)
            cv2.circle(image, (x2, y2), 7, (0, 0, 255), cv2.FILLED)
            cv2.circle(image, (x3, y3), 7, (0, 0, 255), cv2.FILLED)

            cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(image, (x3, y3), (x2, y2), (255, 255, 255), 3)

            angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
            if angle < 0:
                angle += 360
            cv2.putText(image, str(int(angle)), (x2-20, y2+50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            return angle


def main():
    capture = cv2.VideoCapture(0)
    detector = poseDetector()

    while True:
        success, image = capture.read()
        image = detector.drawPose(image)

        # FINAL DISPLAY

        cv2.waitKey(2)
        cv2.imshow("Pose Record", image)


if __name__ == "__main__":
    main()