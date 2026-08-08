from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import tensorflow as tf
import time
from collections import deque, Counter
from datetime import datetime

from utils.hand_detector import HandDetector

# ==========================================
# Flask App
# ==========================================

app = Flask(__name__)

# ==========================================
# Load AI Model
# ==========================================

MODEL_PATH = "model/sign_model.keras"
LABEL_PATH = "model/labels.txt"

model = tf.keras.models.load_model(MODEL_PATH)

with open(LABEL_PATH, "r") as f:
    labels = [line.strip() for line in f.readlines()]

print("✅ Model Loaded Successfully")

# ==========================================
# Hand Detector
# ==========================================

detector = HandDetector()

print("✅ Hand Detector Ready")

# ==========================================
# Camera
# ==========================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("❌ Webcam could not be opened")

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("✅ Webcam Started")

# ==========================================
# Global Variables
# ==========================================

current_prediction = "Waiting..."
current_confidence = 0.0

current_word = ""
current_sentence = ""

prediction_history = deque(maxlen=5)

last_letter = ""
last_added_time = time.time()

# ==========================================
# Settings
# ==========================================

CONFIDENCE_THRESHOLD = 0.80

ADD_DELAY = 1.2

AUTO_SPACE_DELAY = 2.0

# ==========================================
# Helper Functions
# ==========================================

def preprocess_image(roi):
    """
    Resize and normalize ROI before prediction.
    """

    img = cv2.resize(roi, (224, 224))

    img = img.astype(np.float32)

    img = img / 255.0

    img = np.expand_dims(img, axis=0)

    return img


def predict_sign(roi):
    """
    Predict sign using TensorFlow model.
    """

    global prediction_history

    img = preprocess_image(roi)

    prediction = model.predict(img, verbose=0)

    index = np.argmax(prediction)

    confidence = float(prediction[0][index])

    label = labels[index].strip()

    # Low confidence -> Unknown
    if confidence < CONFIDENCE_THRESHOLD:
        return "Unknown", confidence

    # Stable Prediction
    prediction_history.append(label)

    if len(prediction_history) == prediction_history.maxlen:

        label = Counter(prediction_history).most_common(1)[0][0]

    return label, confidence


# ==========================================
# Sentence Builder
# ==========================================

def add_word_to_sentence():

    global current_word
    global current_sentence
    global last_letter

    if current_word.strip() != "":

        current_sentence += current_word + " "

        current_word = ""

        last_letter = ""


# ==========================================
# Clear All
# ==========================================

def clear_all():

    global current_word
    global current_sentence
    global last_letter
    global prediction_history

    current_word = ""

    current_sentence = ""

    last_letter = ""

    prediction_history.clear()


# ==========================================
# Save History
# ==========================================

def save_history():

    if current_sentence.strip() == "":
        return

    with open("history.txt", "a") as file:

        file.write(
            f"{datetime.now().strftime('%d-%m-%Y %H:%M:%S')} : {current_sentence}\n"
        )

    print("✅ History Saved")

# ==========================================
# Video Streaming
# ==========================================

def generate_frames():

    global current_prediction
    global current_confidence
    global current_word
    global current_sentence
    global last_letter
    global last_added_time

    prev_time = time.time()

    print("✅ Video Stream Started")

    while True:

        success, frame = camera.read()

        if not success:
            print("❌ Camera Read Failed")
            continue

        frame = cv2.flip(frame, 1)

        try:

            frame, bbox = detector.detect(frame)

            if bbox is not None:

                x1, y1, x2, y2 = bbox

                h, w = frame.shape[:2]

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)

                roi = frame[y1:y2, x1:x2]

                if roi is not None and roi.size > 0:

                    label, confidence = predict_sign(roi)

                    current_prediction = label
                    current_confidence = confidence

                    valid_label = label.strip().lower()

                    # ==================================
                    # Delete Gesture
                    # ==================================

                    if valid_label == "del":

                        if len(current_word) > 0:
                            current_word = current_word[:-1]

                        last_letter = ""
                        last_added_time = time.time()

                    # ==================================
                    # Space Gesture
                    # ==================================

                    elif valid_label == "space":

                        add_word_to_sentence()

                        last_added_time = time.time()

                    # ==================================
                    # Word Builder
                    # ==================================

                    elif (

                        confidence >= CONFIDENCE_THRESHOLD
                        and valid_label not in ["unknown", "nothing", ""]
                        and valid_label != last_letter
                        and (time.time() - last_added_time) > ADD_DELAY

                    ):

                        current_word += valid_label.upper()

                        last_letter = valid_label

                        last_added_time = time.time()

                    # ==================================
                    # Auto Space
                    # ==================================

                    if (

                        current_word != ""
                        and (time.time() - last_added_time) > AUTO_SPACE_DELAY

                    ):

                        add_word_to_sentence()

                        last_added_time = time.time()

                    # ==================================
                    # Box
                    # ==================================

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    if confidence > 0.95:
                        color = (0, 255, 0)

                    elif confidence > 0.85:
                        color = (0, 255, 255)

                    else:
                        color = (0, 0, 255)

                    cv2.putText(
                        frame,
                        f"{label} ({confidence:.2f})",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        color,
                        2
                    )

            else:

                current_prediction = "No Hand"
                current_confidence = 0

                cv2.putText(
                    frame,
                    "Show Your Hand",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )

        except Exception as e:

            print("Prediction Error:", e)

        # ==================================
        # Current Word
        # ==================================

        cv2.putText(
            frame,
            f"Word : {current_word}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2
        )

        # ==================================
        # Sentence
        # ==================================

        cv2.putText(
            frame,
            f"Sentence : {current_sentence}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        # ==================================
        # FPS
        # ==================================

        current_time = time.time()

        fps = 1 / max(current_time - prev_time, 0.0001)

        prev_time = current_time

        cv2.putText(
            frame,
            f"FPS : {int(fps)}",
            (20, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        ret, buffer = cv2.imencode(".jpg", frame)

        if not ret:
            continue

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            buffer.tobytes() +
            b'\r\n'
        )
# ==========================================
# Home Page
# ==========================================

@app.route("/")
def home():

    return render_template("index.html")


# ==========================================
# Video Stream
# ==========================================

@app.route("/video")
def video():

    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ==========================================
# Prediction API
# ==========================================

@app.route("/prediction")
def prediction():

    return jsonify({

        "label": current_prediction,

        "confidence": round(current_confidence * 100, 2),

        "word": current_word,

        "sentence": current_sentence

    })


# ==========================================
# Clear All
# ==========================================

@app.route("/clear")
def clear():

    clear_all()

    return jsonify({

        "status": "success"

    })


# ==========================================
# Add Current Word to Sentence
# ==========================================

@app.route("/add_word")
def add_word():

    add_word_to_sentence()

    return jsonify({

        "status": "success",

        "sentence": current_sentence

    })
# ==========================================
# Save History
# ==========================================

@app.route("/save")
def save():

    save_history()

    return jsonify({

        "status": "success",

        "message": "History Saved"

    })


# ==========================================
# History Page
# ==========================================

@app.route("/history")
def history():

    try:

        with open("history.txt", "r") as file:

            history_data = file.readlines()

    except FileNotFoundError:

        history_data = []

    history_data.reverse()

    return render_template(

        "history.html",

        history=history_data

    )


# ==========================================
# Health Check
# ==========================================

@app.route("/test")
def test():

    return jsonify({

        "status": "running",

        "camera": camera.isOpened(),

        "prediction": current_prediction,

        "confidence": round(current_confidence * 100, 2)

    })


# ==========================================
# Release Camera
# ==========================================

def release_camera():

    global camera

    if camera.isOpened():

        camera.release()

        print("✅ Camera Released")


# ==========================================
# Run Flask
# ==========================================

if __name__ == "__main__":

    try:

        app.run(

            host="127.0.0.1",

            port=5001,

            debug=False,

            threaded=True

        )

    finally:

        release_camera()                