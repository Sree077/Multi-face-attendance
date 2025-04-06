import os
import face_recognition
import cv2
import pickle
import pandas as pd
from datetime import datetime

ENCODINGS_FILE = "encodings.pkl"
ATTENDANCE_FILE = "attendance.csv"

# Load saved encodings
with open(ENCODINGS_FILE, "rb") as file:
    data = pickle.load(file)
    known_encodings = data["encodings"]
    known_names = data["names"]

# Initialize webcam
cap = cv2.VideoCapture(0)

# Create an attendance CSV if not exists
if not os.path.exists(ATTENDANCE_FILE):
    df = pd.DataFrame(columns=["Name", "Time"])
    df.to_csv(ATTENDANCE_FILE, index=False)

recognized_students = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
        name = "Unknown"

        if True in matches:
            match_index = matches.index(True)
            name = known_names[match_index]

            # Mark attendance
            if name not in recognized_students:
                recognized_students.add(name)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                df = pd.read_csv(ATTENDANCE_FILE)
                df = pd.concat([df, pd.DataFrame([[name, timestamp]], columns=["Name", "Time"])], ignore_index=True)
                df.to_csv(ATTENDANCE_FILE, index=False)
                print(f"Attendance marked for {name}")

        # Draw rectangle and label
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Face Recognition Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
