# Import necessary libraries
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
import cv2
import numpy as np

# Load pre-trained Haar Cascade classifier for face detection
face_classifier = cv2.CascadeClassifier(
    './haarcascade_frontalface_default.xml')

# Load pre-trained emotion detection model
classifier = load_model('./Emotion_Detection.h5')

# Define emotion labels
class_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Specify the path to the image you want to test
image_path = 'facial-expressions.jpg'

# Load the image
frame = cv2.imread(image_path)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Detect faces in the image using Haar Cascade
faces = face_classifier.detectMultiScale(gray, 1.3, 5)

# Loop over each detected face
for (x, y, w, h) in faces:
    # Draw a rectangle around the detected face
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Extract the region of interest (ROI) and resize it
    roi_gray = gray[y:y + h, x:x + w]
    roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

    # Preprocess the ROI for the emotion detection model
    if np.sum([roi_gray]) != 0:
        roi = roi_gray.astype('float') / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        # Make predictions using the emotion detection model
        preds = classifier.predict(roi)[0]
        label = class_labels[preds.argmax()]
        label_position = (x, y)

        # Reduce the font size by changing the fontScale parameter
        font_scale = 0.8

        # Display the predicted emotion label on the image
        cv2.putText(frame, label, label_position,
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)
    else:
        # If no face is found in the ROI, display a message
        cv2.putText(frame, 'No Face Found', (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

# Display the image with emotion predictions
cv2.imshow('Emotion Detector', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
