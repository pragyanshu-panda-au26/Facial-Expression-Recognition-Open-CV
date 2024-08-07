import numpy as np
import cv2
import streamlit as st
from tensorflow import keras
from keras.models import model_from_json, Sequential
from keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, WebRtcMode
import json
import os

# Load model
emotion_dict = {0: 'angry', 1: 'happy', 2: 'neutral', 3: 'sad', 4: 'surprise'}

# Load json and create model
json_file = open('emotion_model1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
classifier = model_from_json(loaded_model_json, custom_objects={'Sequential': Sequential})

# Check if the file exists
if not os.path.exists("model1.h5"):
    raise FileNotFoundError("The model1.h5 file was not found.")

# Load weights into new model
classifier.load_weights("model1.h5")

# Load face cascade
try:
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
except Exception as e:
    st.write(f"Error loading cascade classifiers: {e}")

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

class Faceemotion(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Convert image to gray
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = img_gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                prediction = classifier.predict(roi)[0]
                maxindex = int(np.argmax(prediction))
                finalout = emotion_dict[maxindex]
                output = str(finalout)
                label_position = (x, y)
                cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img

def main():
    st.title("Real Time Face Emotion Detection Application")
    activities = ["Home", "Webcam Face Detection", "About"]
    choice = st.sidebar.selectbox("Select Activity", activities)
    st.sidebar.markdown(
        """ Developed by Prag   
            Email : prah.dev@protonmail.com  
        """
    )

    if choice == "Home":
        html_temp_home1 = """<div style="background-color:#6D7B8D;padding:10px">
                                <h4 style="color:white;text-align:center;">
                                Face Emotion detection application using OpenCV, Custom CNN model and Streamlit.</h4>
                             </div>
                             </br>"""
        st.markdown(html_temp_home1, unsafe_allow_html=True)
        st.write("""
                 The application has two functionalities.

                 1. Real-time face detection using webcam feed.

                 2. Real-time face emotion recognition.
                 """)
    elif choice == "Webcam Face Detection":
        st.header("Webcam Live Feed")
        st.write("Click on start to use webcam and detect your face emotion")
        webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION,
                        video_processor_factory=Faceemotion)
    elif choice == "About":
        st.subheader("About this app")
        html_temp_about1 = """<div style="background-color:#6D7B8D;padding:10px">
                                <h4 style="color:white;text-align:center;">
                                Real-time face emotion detection application using OpenCV, Custom Trained CNN model and Streamlit.</h4>
                              </div>
                              </br>"""
        st.markdown(html_temp_about1, unsafe_allow_html=True)

        html_temp4 = """
                      <div style="background-color:#98AFC7;padding:10px">
                      <h4 style="color:white;text-align:center;">
                      This Application is developed by Mohammad Juned Khan using Streamlit Framework, Opencv, Tensorflow and Keras library for demonstration purposes. If you're on LinkedIn and want to connect, just click on the link in the sidebar and shoot me a request. If you have any suggestions or want to comment, just write a mail to Mohammad.juned.z.khan@gmail.com.</h4>
                      <h4 style="color:white;text-align:center;">Thanks for Visiting</h4>
                      </div>
                      <br></br>
                      <br></br>"""
        st.markdown(html_temp4, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
