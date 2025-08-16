import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('traffic_sign_model.keras')

# Initialize webcam or video input
cap = cv2.VideoCapture(0)  # 0 for webcam, replace with video file path for video

# Image resizing parameters
img_size = 64

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    frame_resized = cv2.resize(frame, (img_size, img_size))
    frame_resized = frame_resized / 255.0
    frame_resized = np.expand_dims(frame_resized, axis=0)  # Add batch dimension

    # Predict the traffic sign
    prediction = model.predict(frame_resized)
    predicted_class = np.argmax(prediction)

    # Display the prediction on the frame
    cv2.putText(frame, f'Predicted: {predicted_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Traffic Sign Recognition", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
