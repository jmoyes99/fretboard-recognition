import cv2
import numpy as np
import tensorflow as tf

# load the TFLite model
interpreter = tf.lite.Interpreter(model_path="C:/111Programming/fretboard-recognition/model/converted_tflite/model.tflite")
interpreter.allocate_tensors()

# required step for tensor input and output
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# chord labels
with open("C:/111Programming/fretboard-recognition/model/converted_tflite/labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# setting up webcam
cap = cv2.VideoCapture(1)

# general initialisation
def preprocess(frame):
    img = cv2.resize(frame, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# primary loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

# sends images to model and predicts likelihood of chords
    input_data = preprocess(frame)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]

# interprets results
    predicted_index = np.argmax(output_data)
    confidence = output_data[predicted_index]
    label = f"{labels[predicted_index]} ({confidence*100:.2f}%)"

# displays chord names
    cv2.putText(frame, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("Guitar Chord Recognition", frame)

# ends program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# closes webcam
cap.release()
cv2.destroyAllWindows()
