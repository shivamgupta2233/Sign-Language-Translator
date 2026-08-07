from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import tensorflow as tf
from utils.hand_detector import HandDetector
import time

# ==========================
# Flask App
# ==========================

app = Flask(__name__)

# ==========================
# Load AI Model
# ==========================

model = tf.keras.models.load_model("model/sign_model.keras")

with open("model/labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# ==========================
# Hand Detector
# ==========================

detector = HandDetector()

# ==========================
# Webcam
# ==========================

camera = cv2.VideoCapture(0)

# ==========================
# Global Variables
# ==========================

current_prediction = "None"
current_confidence = 0.0
current_word = ""
sentence = ""
last_letter = ""

# ==========================
# Video Stream
# ==========================

def generate_frames():

    global current_prediction
    global current_confidence
    last_added_time = time.time()
    ADD_DELAY = 1.5

    while True:

        success, frame = camera.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)

        # Detect Hand
        frame, bbox = detector.detect(frame)

        if bbox is not None:

            x1, y1, x2, y2 = bbox

            roi = frame[y1:y2, x1:x2]

            if roi.size != 0:

                img = cv2.resize(roi, (224, 224))
                img = img.astype(np.float32) / 255.0
                img = np.expand_dims(img, axis=0)

                prediction = model.predict(img, verbose=0)

                index = np.argmax(prediction)

                confidence = float(prediction[0][index])

                label = labels[index]

                if confidence < 0.80:
                    label = "Unknown"

                current_prediction = label
                current_confidence = confidence
                global current_word
                global last_letter
                global last_added_time

                if (
                    label not in ["Unknown", "nothing"]
                     and label != last_letter
                     and (time.time() - last_added_time) > ADD_DELAY
                    ):
                 current_word += label
                 last_letter = label
                 last_added_time = time.time()

                print(f"Prediction: {label} ({confidence:.2f})")

                cv2.putText(
                    frame,
                    f"{label} ({confidence:.2f})",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

        else:

            cv2.putText(
                frame,
                "Show your hand",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

        ret, buffer = cv2.imencode(".jpg", frame)

        frame = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            frame +
            b'\r\n'
        )

# ==========================
# Routes
# ==========================

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/video")
def video():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/prediction")
def prediction():

    return jsonify({
    "label": current_prediction,
    "confidence": round(current_confidence * 100, 2),
    "word": current_word,
    "sentence": sentence
})
# ==========================
# Run
# ==========================

if __name__ == "__main__":
    app.run(debug=True)