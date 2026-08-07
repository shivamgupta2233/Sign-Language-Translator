import cv2
import numpy as np
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("model/sign_model.keras")

# Load labels
with open("model/labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit()

print("✅ Webcam Started")
print("Press Q to Exit")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Mirror image
    frame = cv2.flip(frame, 1)

    # ROI (Region of Interest)
    x1, y1 = 150, 100
    x2, y2 = 450, 400

    roi = frame[y1:y2, x1:x2]

    # Preprocess
    img = cv2.resize(roi, (224, 224))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img, verbose=0)

    index = np.argmax(prediction)
    confidence = prediction[0][index]

    label = labels[index]

    # Draw ROI
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

    cv2.putText(
        frame,
        f"{label} ({confidence:.2f})",
        (20,50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.imshow("AI Sign Language Translator", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()