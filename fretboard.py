import sqlite3
import face_recognition
import cv2
import numpy as np
 
# Connect to the SQLite database
conn = sqlite3.connect("face_database.db")
cursor = conn.cursor()

# Load known face encodings from database
cursor.execute("SELECT name, encoding FROM faces")
known_faces = cursor.fetchall()
known_names = []
known_encodings = []
 
for name, encoding in known_faces:
    known_names.append(name)
    known_encodings.append(np.frombuffer(encoding, dtype=np.float64))
 
# Open a connection to the webcam
video_capture = cv2.VideoCapture(1)
 
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
 
 
while True:
    # Capture a frame
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to grab frame")
        break
 
    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
 
    # Detect faces in the frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
 
    for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"
 
        # Find best match
        if True in matches:
            match_index = matches.index(True)
            name = known_names[match_index]
 
        # Draw rectangle and name
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
 
    # Display the frame
    cv2.imshow("Face Recognition", frame)
 
    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
 
# Release webcam and close windows
video_capture.release()
cv2.destroyAllWindows()
conn.close()