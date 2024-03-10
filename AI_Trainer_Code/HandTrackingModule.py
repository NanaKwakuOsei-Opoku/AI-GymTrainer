import cv2
import mediapipe as mp
import time


class HandDetector():
    # Attributes of the class
    def __init__(self, mode=False, max_hands=2, detect_con=0.5, track_con=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detect_con = detect_con
        self.track_con = track_con
        self.mpDraw = mp.solutions.drawing_utils
        self.mpHands = mp.solutions.hands
        self.MyHand = self.mpHands.Hands(self.mode, self.max_hands, self.detect_con, self.track_con)

    def drawLandmarks(self, image):
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.MyHand.process(imageRGB)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(image, hand_landmarks, self.mpHands.HAND_CONNECTIONS)

        return image

    def printPosition(self, image):
        positions = []

        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.MyHand.process(imageRGB)
        height, width, _ = image.shape

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for id, landmark in enumerate(hand_landmarks.landmark):
                    x, y = int(landmark.x * height), int(landmark.y * width)
                    positions.append((id, x, y))

                    # Print position to console
                    print(f"Landmark {id}: ({x}, {y})")

def main():
    capture = cv2.VideoCapture(0)
    detector = HandDetector() # Creating an instance of the handDetector class
    prev_time,current_time = 0,0

    while True:
        success, image = capture.read()
        image = detector.drawLandmarks(image)
        positions = detector.printPosition(image)
        print(positions)

        # FrameRate-
        current_time = time.time()
        frame_rate = 1 / (current_time - prev_time)
        prev_time = current_time
        cv2.putText(image, str(int(frame_rate)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # FINAL DISPLAY
        cv2.imshow("Hand Tracking", image)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
