import pyttsx3
import cv2
import numpy as np
import tensorflow as tf
from collections import deque, Counter
from utils.hand_detector import HandDetector

# ==========================
# Load Model & Labels
# ==========================

model = tf.keras.models.load_model("model/sign_model.keras")

with open("model/labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# ==========================
# Initialize
# ==========================

detector = HandDetector()
engine = pyttsx3.init()
engine.setProperty("rate", 150)
prediction_history = deque(maxlen=5)

current_word = ""
last_letter = ""

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit()

print("✅ Webcam Started")
print("Press Q to Exit")

# ==========================
# Main Loop
# ==========================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    frame, bbox = detector.detect(frame)

    # --------------------------
    # No Hand Detected
    # --------------------------

    if bbox is None:

        cv2.putText(
            frame,
            "Show your hand",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

        cv2.putText(
            frame,
            f"Word: {current_word}",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
        )

        cv2.imshow("AI Sign Language Translator", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("c"):
            current_word = ""
            last_letter = ""

        elif key == ord(" "):
            current_word += " "

        elif key == 8:
            current_word = current_word[:-1]

        continue

    # --------------------------
    # Crop Hand
    # --------------------------

    x1, y1, x2, y2 = bbox

    roi = frame[y1:y2, x1:x2]

    if roi.size == 0:
        continue

    img = cv2.resize(roi, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    # --------------------------
    # Prediction
    # --------------------------

    prediction = model.predict(img, verbose=0)

    index = np.argmax(prediction)

    confidence = float(prediction[0][index])

    label = labels[index]

    # Confidence Threshold

    if confidence < 0.80:
        label = "Unknown"

    # Stable Prediction

    prediction_history.append(label)

    if len(prediction_history) == 5:
        label = Counter(prediction_history).most_common(1)[0][0]

    # Word Builder

    if label not in ["Unknown", "nothing"]:

        if label != last_letter:

            current_word += label

            last_letter = label
    
    cv2.putText(
    frame,
    "Q:Quit  C:Clear  S:Speak",
    (20, 140),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (255,255,255),
    2
)        

    # --------------------------
    # Display
    # --------------------------

    cv2.putText(
        frame,
        f"{label} ({confidence:.2f})",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        frame,
        f"Word: {current_word}",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2,
    )

    cv2.imshow("AI Sign Language Translator", frame)

    # --------------------------
    # Keyboard
    # --------------------------

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    elif key == ord("c"):
        current_word = ""
        last_letter = ""

    elif key == ord(" "):
        current_word += " "

    elif key == 8:
        current_word = current_word[:-1]
    
    elif key == ord("s"):

       if current_word.strip() != "":
        engine.say(current_word)
        engine.runAndWait()    

cap.release()
cv2.destroyAllWindows()