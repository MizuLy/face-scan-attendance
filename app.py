"""
Task 2: AI Attendance via Camera (Face Recognition)

Uses OpenCV's built-in LBPH face recognizer (no dlib needed - easy
install on Windows). Webcam capture happens in the browser (JS);
the captured photo is sent here for face detection + recognition.

Setup:
    1. Put one clear front-facing photo per person in known_faces/,
       named after them, e.g. known_faces/sengly.jpg
    2. pip install opencv-contrib-python flask
    3. python3 app.py
    4. Open http://127.0.0.1:8000

Attendance is logged to attendance_log.csv (auto-created), one
entry per person per day (won't log the same person twice in a day).
"""

import os
import csv
import base64
from datetime import datetime
import cv2
import numpy as np
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

KNOWN_FACES_DIR = "known_faces"
LOG_FILE = "attendance_log.csv"
FACE_SIZE = (200, 200)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
recognizer = cv2.face.LBPHFaceRecognizer_create()

label_to_name = {}  # int label -> person name


def detect_face(gray_img):
    """Return the largest detected face region, or None."""
    faces = face_cascade.detectMultiScale(
        gray_img, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return None
    # pick the largest face box
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face = gray_img[y:y + h, x:x + w]
    return cv2.resize(face, FACE_SIZE)


def train_from_known_faces():
    """Load known_faces/*.jpg, detect a face in each, train the recognizer."""
    global label_to_name
    faces, labels = [], []
    label_to_name = {}

    if not os.path.isdir(KNOWN_FACES_DIR):
        os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

    files = [f for f in os.listdir(KNOWN_FACES_DIR)
             if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    for idx, filename in enumerate(files):
        name = os.path.splitext(filename)[0]
        path = os.path.join(KNOWN_FACES_DIR, filename)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        face = detect_face(img)
        if face is None:
            print(f"[warn] no face found in {filename}, skipping")
            continue
        faces.append(face)
        labels.append(idx)
        label_to_name[idx] = name

    if faces:
        recognizer.train(faces, np.array(labels))
        print(
            f"Trained on {len(faces)} known face(s): {list(label_to_name.values())}")
    else:
        print("No known faces trained yet. Add photos to known_faces/ and restart.")

    return len(faces)


def log_attendance(name):
    """Append a row to the CSV, skipping if already logged today."""
    today = datetime.now().strftime("%Y-%m-%d")
    now = datetime.now().strftime("%H:%M:%S")

    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, newline="") as f:
            for row in csv.reader(f):
                if len(row) >= 2 and row[0] == name and row[1] == today:
                    return False  # already checked in today

    write_header = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["name", "date", "time"])
        writer.writerow([name, today, now])
    return True


@app.route("/")
def index():
    return render_template("index.html", known_count=len(label_to_name))


@app.route("/check_in", methods=["POST"])
def check_in():
    data = request.json.get("image", "")
    if not data.startswith("data:image"):
        return jsonify({"status": "error", "message": "No image received."})

    if not label_to_name:
        return jsonify({"status": "error",
                        "message": "No known faces trained. Add photos to known_faces/ and restart the server."})

    # Decode base64 image from the browser
    header, encoded = data.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face = detect_face(gray)
    if face is None:
        return jsonify({"status": "no_face", "message": "No face detected. Try again."})

    label, distance = recognizer.predict(face)
    # LBPH: lower distance = better match. ~<70 is usually a decent match.
    CONFIDENCE_THRESHOLD = 70

    if distance < CONFIDENCE_THRESHOLD:
        name = label_to_name[label]
        confidence_pct = max(0, round(100 - distance, 1))
        logged = log_attendance(name)
        msg = f"Welcome, {name}!" if logged else f"{name} already checked in today."
        return jsonify({"status": "recognized", "name": name,
                        "confidence": confidence_pct, "logged": logged, "message": msg})
    else:
        return jsonify({"status": "unknown",
                        "message": "Face not recognized. Are you in known_faces/?"})


if __name__ == "__main__":
    train_from_known_faces()
    app.run(host="127.0.0.1", port=8000, debug=True, use_reloader=False)
