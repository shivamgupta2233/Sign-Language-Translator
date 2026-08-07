import cv2
import mediapipe as mp
import os
import time

# ===== User Input =====
LETTER = input("Enter Letter (A-Z): ").upper()

if len(LETTER) != 1 or not LETTER.isalpha():
    print("❌ Please enter a single letter from A to Z.")
    exit()

TOTAL_IMAGES = 500

SAVE_PATH = f"dataset/{LETTER}"
os.makedirs(SAVE_PATH, exist_ok=True)

# ===== MediaPipe =====
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# ===== Camera =====
cap = cv2.VideoCapture(0)

count = len(os.listdir(SAVE_PATH))

print(f"\nCollecting dataset for Letter: {LETTER}")
print("Starting in 3 seconds...")
time.sleep(3)

last_save = time.time()

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    hand_detected = False

    if results.multi_hand_landmarks:
        hand_detected = True

        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    cv2.putText(frame, f"Letter : {LETTER}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.putText(frame, f"Saved : {count}/{TOTAL_IMAGES}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    if hand_detected and (time.time() - last_save > 0.1):
        filename = os.path.join(SAVE_PATH, f"{count}.jpg")
        cv2.imwrite(filename, frame)
        count += 1
        last_save = time.time()

    cv2.imshow("Dataset Collection", frame)

    if count >= TOTAL_IMAGES:
        print("✅ Dataset Collection Completed!")
        break

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()