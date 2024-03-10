import cv2
import time
import HandTrackingModule as htm

capture = cv2.VideoCapture(0)
detector = htm.HandDetector()
prev_time,current_time = 0,0

while True:
    success, image = capture.read()
    image = detector.drawLandmarks(image)
    positions = detector.printPosition(image)
    if positions:
        print(positions[0])

    # FrameRate
    current_time = time.time()  # this returns the current time in secs as floating-point value
    frame_rate = 1 / (current_time - prev_time)
    prev_time = current_time
    cv2.putText(image, str(int(frame_rate)), (100, 100), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    # FINAL DISPLAY
    cv2.waitKey(1)
    cv2.imshow("Capturing Nana Kwaku's Hand.....", image)








