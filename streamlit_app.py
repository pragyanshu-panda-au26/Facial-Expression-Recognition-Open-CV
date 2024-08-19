import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import mediapipe as mp

# Load the face classifier and emotion detection model
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
classifier = load_model('modelv1.h5')
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Initialize Mediapipe face mesh solution
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

class EmotionDetector(VideoProcessorBase):
    def __init__(self):
        self.face_classifier = face_classifier
        self.classifier = classifier
        self.emotion_labels = emotion_labels
        self.face_mesh = face_mesh

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        try:
            img = frame.to_ndarray(format="bgr24")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
                roi_gray = gray[y:y + h, x:x + w]
                roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

                if np.sum([roi_gray]) != 0:
                    roi = roi_gray.astype('float') / 255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)

                    prediction = self.classifier.predict(roi)[0]
                    label = self.emotion_labels[prediction.argmax()]
                    label_position = (x, y)
                    cv2.putText(img, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # Convert the region of interest to RGB for Mediapipe processing
                    roi_rgb = cv2.cvtColor(img[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
                    results = self.face_mesh.process(roi_rgb)

                    # Draw the facial landmarks using Mediapipe
                    if results.multi_face_landmarks:
                        for face_landmarks in results.multi_face_landmarks:
                            for landmark in face_landmarks.landmark:
                                x_point = int(landmark.x * w) + x
                                y_point = int(landmark.y * h) + y
                                cv2.circle(img, (x_point, y_point), 1, (0, 0, 255), -1)

        except Exception as e:
            st.error(f"Error processing frame: {e}")

        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.title("Facial Emotion Recognition")
st.write("This application detects facial emotions in real-time from your webcam feed.")

webrtc_streamer(key="emotion-detector", video_processor_factory=EmotionDetector)
