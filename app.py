import cv2
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load a pre-trained emotion recognition model (this is just an example, you need a model file)
emotion_model = load_model('model.h5')

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load music data
Music_Player = pd.read_csv("data_moods.csv")
Music_Player = Music_Player[['name', 'artist', 'mood', 'popularity']]

# Function to detect emotions
def detect_emotions(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
    predicted_emotion = ""
    for (x, y, w, h) in faces:
        roi_gray = gray_frame[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))  # Resize to the size expected by the model
        roi_gray = roi_gray.astype('float32') / 255  # Normalize the image
        roi_gray = roi_gray.reshape(1, 48, 48, 1)

        # Predict the emotion
        predictions = emotion_model.predict(roi_gray)
        max_index = predictions[0].argmax()
        predicted_emotion = emotion_labels[max_index]
    return predicted_emotion

# Function to recommend songs based on emotion
def Recommend_Songs(pred_class):
    if pred_class == 'Disgust':
        Play = Music_Player[Music_Player['mood'] == 'Sad']
    elif pred_class in ['Happy', 'Sad']:
        Play = Music_Player[Music_Player['mood'] == 'Happy']
    elif pred_class in ['Fear', 'Angry']:
        Play = Music_Player[Music_Player['mood'] == 'Calm']
    elif pred_class in ['Surprise', 'Neutral']:
        Play = Music_Player[Music_Player['mood'] == 'Energetic']
    else:
        return None

    Play = Play.sort_values(by="popularity", ascending=False)[:5].reset_index(drop=True)
    return Play

# Streamlit App
st.title("Emotion-Based Music Recommendation")

# Webcam-based emotion detection
if 'run' not in st.session_state:
    st.session_state.run = False

if st.button("Start Emotion Detection"):
    st.session_state.run = True

if st.session_state.run:
    cap = cv2.VideoCapture(0)
    emotion_list = []
    stframe = st.empty()

    stop_detection = st.button("Stop Detection")

    while cap.isOpened() and st.session_state.run:
        ret, frame = cap.read()
        if not ret:
            st.warning("Unable to access webcam. Please check your device.")
            break

        # Detect emotions
        detected_emotion = detect_emotions(frame)
        emotion_list.append(detected_emotion)

        # Display the webcam feed with detected emotion
        frame = cv2.putText(frame, detected_emotion, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        # Check for stop button click
        if stop_detection:
            st.session_state.run = False
            break

    cap.release()

    # Determine dominant emotion
    if emotion_list:
        dominant_emotion = max(set(emotion_list), key=emotion_list.count)
        st.subheader(f"Detected Emotion: {dominant_emotion}")

        # Recommend songs based on emotion
        st.subheader("Recommended Songs:")
        songs = Recommend_Songs(dominant_emotion)
        if songs is not None:
            st.table(songs)
        else:
            st.write("No songs found for the detected emotion.")

# Instructions
st.write("Click 'Stop Detection' to stop the webcam feed.")
