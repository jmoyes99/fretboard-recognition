"""
This Guitar Chord recognition tool uses Google's Teachable Machine and TensorFlow Lite.

The program uses a webcam to capture real-time video, processing and analysing frames 
to classify chords according to a premade database. The pretrained model predicts the 
chance percentage of a particular chord being on screen.

Press run to begin, and 'q' to quit the application.
"""
import cv2
import numpy as np
import tensorflow as tf

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="C:/111Programming/fretboard-recognition/model/converted_tflite/model.tflite")
interpreter.allocate_tensors()

# This is a required step for tensor input and output
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Loads chord labels
with open("C:/111Programming/fretboard-recognition/model/converted_tflite/labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Setting up webcam
cap = cv2.VideoCapture(1)

# General initialisation
def preprocess(frame):
    img = cv2.resize(frame, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Primary loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

# Sends images to model and predicts likelihood of chords
    input_data = preprocess(frame)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]

# Interprets results
    predicted_index = np.argmax(output_data)
    confidence = output_data[predicted_index]
    label = f"{labels[predicted_index]} ({confidence*100:.2f}%)"

# Displays chord names
    cv2.putText(frame, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("Guitar Chord Recognition", frame)

# Ends program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Closes webcam
cap.release()
cv2.destroyAllWindows()
