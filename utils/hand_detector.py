import cv2
import mediapipe as mp

class HandDetector:

    def __init__(self):
        self.mp_hands = mp.solutions.hands

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        self.mp_draw = mp.solutions.drawing_utils

    def detect(self, frame):

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.hands.process(rgb)

        bbox = None

        if results.multi_hand_landmarks:

            for hand in results.multi_hand_landmarks:

                self.mp_draw.draw_landmarks(
                    frame,
                    hand,
                    self.mp_hands.HAND_CONNECTIONS
                )

                h, w, _ = frame.shape

                xs = []
                ys = []

                for lm in hand.landmark:
                    xs.append(int(lm.x * w))
                    ys.append(int(lm.y * h))

                x1 = max(min(xs) - 20, 0)
                y1 = max(min(ys) - 20, 0)

                x2 = min(max(xs) + 20, w)
                y2 = min(max(ys) + 20, h)

                bbox = (x1, y1, x2, y2)

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

        return frame, bbox