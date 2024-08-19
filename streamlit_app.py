import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, VideoProcessorBase
import av
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import dlib
from imutils import face_utils

# Load the face detector and facial landmark predictor from dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
classifier = load_model('modelv1.h5')
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

class EmotionDetector(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                prediction = classifier.predict(roi)[0]
                label = emotion_labels[prediction.argmax()]
                label_position = (x, y)
                cv2.putText(img, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Convert the ROI to a dlib rectangle
                rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                # Draw the facial landmarks
                for (i, (x_point, y_point)) in enumerate(shape):
                    cv2.circle(img, (x_point, y_point), 1, (0, 0, 255), -1)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.title("Facial Emotion Recognition")
st.write("This application detects facial emotions in real-time from your webcam feed.")

webrtc_streamer(key="emotion-detector", video_processor_factory=EmotionDetector)
