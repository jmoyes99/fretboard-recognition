import sqlite3
import face_recognition 
import cv2
import numpy as np
 
# Connect to SQLite database (or create it)
conn = sqlite3.connect("face_database.db")
cursor = conn.cursor()
 
# Create table to store face encodings
cursor.execute("""
CREATE TABLE IF NOT EXISTS faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    encoding BLOB NOT NULL
)
""")
conn.commit() # Complex technique
 
def encode_face(image_path, name):
    """Encodes a face and stores it in the database."""
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)
    
    if len(encodings) > 0:
        encoding = encodings[0]  # Get first encoding
        encoding_bytes = np.array(encoding).tobytes()  # Convert to binary
 
        # Insert into database
        cursor.execute("INSERT INTO faces (name, encoding) VALUES (?, ?)", (name, encoding_bytes))
        conn.commit()
        print(f"Stored {name}'s face in database.")
    else:
        print(f"No face found in {image_path}")
 
 
 
 
 
 
# Add known face
# list of known faces
faces = [
    ("fretboard-recognition/Faces/James/James1.jpg", "James"),
    ("fretboard-recognition/Faces/Noah/Noah1.jpg", "Noah"),
    ("fretboard-recognition/Faces/Veeran/Veeran1.jpg", "Veeran"),
]
 
for image_path, name in faces:
    encode_face(image_path, name)
 
 
conn.close()