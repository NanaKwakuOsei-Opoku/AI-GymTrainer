import cv2
import numpy
import time
from PoseModule import poseDetector

capture = cv2.VideoCapture(0)
detector = poseDetector()
current_time, prev_time = 0, time.time()

# Move these outside the loop
direction = 0
count = 0

while True:
    success, image = capture.read()
    image = detector.drawPose(image)
    positions = detector.return_Positions(image)

    # LEFT ARM
    angle = detector.findAngle(image, 12, 14, 16)

    # Ensure angle is not None and within the valid range
    if angle is not None and 27 <= angle <= 173:
        percentCurl = numpy.interp(angle, (27, 173), (100, 0))
        bar = numpy.interp(angle, (27, 173), (100, 650))  # Adjusted range to cover entire height

        # When the Arm completes one full rep
        if direction == 0 and percentCurl > 95:  # Arm fully curled
            count += 1
            direction = 1
        elif direction == 1 and percentCurl < 5:  # Arm fully extended
            direction = 0
        print(percentCurl, count)

        # Drawing the Bar
        cv2.rectangle(image, (1100, 100), (1175, 650), (255, 255, 0), 3)
        cv2.rectangle(image, (1100, int(bar)), (1175, 650), (255, 255, 0), cv2.FILLED)
        cv2.putText(image, f'{int(percentCurl)}%', (1100, 75), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)

    # Drawing the Curl Count
    cv2.rectangle(image, (5, 5), (220, 220), (255, 255, 0), cv2.FILLED)
    cv2.putText(image, str(int(count)), (15, 160), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 0), 15)

    # FrameRate
    current_time = time.time()
    frame_rate = 1 / (current_time - prev_time)
    prev_time = current_time

    # FINAL DISPLAY
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    cv2.imshow("Pose Record", image)
