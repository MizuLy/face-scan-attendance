# AI Attendance via Camera

A web app that checks people in using face recognition via webcam.

## What it does
Click "Check In," it captures your face through the webcam, matches it
against known people, and logs the name + date + time to a CSV file.
Won't log the same person twice on the same day.

## How it works
- **Face detection:** OpenCV Haar Cascade (finds the face in the frame)
- **Face recognition:** OpenCV's built-in LBPH recognizer (matches the
  detected face against known people)
- Uses `opencv-contrib-python` instead of the more common `face_recognition`
  library, because that one needs `dlib`, which is painful to install on
  Windows (needs C++ build tools). This version installs cleanly with pip.
- Runs 100% locally — camera feed and recognition never leave your computer.

## Folder structure
```
task2_attendance/
├── app.py
├── README.md
├── known_faces/          <- add one photo per person here
│   ├── John Doe.jpg
│   └── friend1.jpg
├── templates/
│   └── index.html
└── attendance_log.csv    <- auto-created after first check-in
```

## Setup
```bash
pip install opencv-contrib-python flask
```

## Add known faces
Put **one clear, front-facing photo per person** in `known_faces/`.
The filename (without extension) becomes their recognized name.

Example: `known_faces/John Doe.jpg` -> recognized as "John Doe"

## Run it
```bash
python app.py
```
Terminal should print something like:
```
Trained on 2 known face(s): ['John Doe', 'friend1']
```
That confirms the photos loaded correctly.

Then open **http://127.0.0.1:8000** in your browser. Allow camera access
when asked, then click "Check In."

## Attendance log
`attendance_log.csv` is created automatically after the first successful
check-in. Columns: `name, date, time`.

## Known limitations
- Needs decent lighting and a front-facing angle to detect a face.
- Works best with one person in frame at a time.
- LBPH recognition is a simpler/older technique than deep-learning face
  recognition (like the `face_recognition` library) — it's less accurate
  on look-alikes or poor lighting, but far easier to install and good
  enough for a class demo.
- If "Face not recognized" keeps showing for someone who IS in
  `known_faces/`, try a clearer, better-lit reference photo.

## Credits
Built with Flask + OpenCV (`opencv-contrib-python`).
