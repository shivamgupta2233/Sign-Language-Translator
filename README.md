# 🤟 AI Sign Language Translator

An AI-powered Sign Language Translator that detects hand gestures using a webcam and translates them into text in real time.

---

## 📌 Features

- 📷 Live Webcam Detection
- ✋ Real-Time Hand Detection (MediaPipe)
- 🤖 AI Sign Prediction (TensorFlow)
- 🔤 Word Builder
- 📝 Sentence Builder
- 📊 Confidence Score
- 📈 Progress Bar
- 🔊 Text-to-Speech
- 💾 Save Translation History
- 📜 View History
- 🎨 Modern Responsive UI

---

## 🛠 Tech Stack

- Python
- Flask
- TensorFlow / Keras
- OpenCV
- MediaPipe
- HTML
- CSS
- JavaScript

---

## 📂 Project Structure

```text
Sign-Language-Translator/
│
├── app.py
├── predict.py
├── collect_dataset.py
├── train_model.py
├── requirements.txt
├── README.md
│
├── model/
│   ├── sign_model.keras
│   └── labels.txt
│
├── templates/
│   ├── index.html
│   └── history.html
│
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── script.js
│
├── utils/
│   ├── hand_detector.py
│   └── preprocessing.py
│
└── history.txt
```

---

## ⚙ Installation

### Clone Repository

```bash
git clone https://github.com/shivamgupta2233/Sign-Language-Translator.git
```

### Open Project

```bash
cd Sign-Language-Translator
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

#### macOS/Linux

```bash
source venv/bin/activate
```

#### Windows

```bash
venv\Scripts\activate
```

### Install Requirements

```bash
pip install -r requirements.txt
```

---

## ▶ Run Project

```bash
python app.py
```

Open in browser:

```text
http://127.0.0.1:5001
```

---

## 🚀 Future Improvements

- 🌍 Multi-language Translation
- 🤖 AI Sentence Auto-Correction
- ☁ Cloud Deployment
- 📱 Mobile Support
- 👤 User Login System
- 💾 Database Integration
- 📄 Export History to PDF

---

## 👨‍💻 Author

**Shivam Kumar**

GitHub:
https://github.com/shivamgupta2233

---

## 📄 License

This project is created for learning and educational purposes.