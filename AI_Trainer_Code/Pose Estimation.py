
import cv2
import time
from PoseModule import poseDetector

capture = cv2.VideoCapture('IMG_1873 2.m4v')
detector = poseDetector()
current_time, prev_time = 0, time.time()

while True:
    success, image = capture.read()
    image = detector.drawPose(image)
    positions = detector.return_Positions(image)





    # FrameRate
    current_time = time.time()
    frame_rate = 1 / (current_time - prev_time)
    prev_time = current_time
    cv2.putText(image, str(int(frame_rate)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # FINAL DISPLAY
    cv2.waitKey(2)
    cv2.imshow("Pose Record", image)






