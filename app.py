import streamlit as st
import whisper
import cv2
import numpy as np
import mediapipe as mp
import time
import pandas as pd
from transformers import pipeline, AutoTokenizer
import winsound
from datetime import timedelta

st.set_page_config(layout="wide")

# Load models
@st.cache_resource
def load_models():
    whisper_model = whisper.load_model("base")
    qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")
    return whisper_model, qa_pipeline, tokenizer

whisper_model, qa_pipeline, tokenizer = load_models()

# Mediapipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1)

# EAR calculation
def calculate_ear(landmarks, eye_indices):
    left = np.array(landmarks[eye_indices[0]])
    right = np.array(landmarks[eye_indices[3]])
    top = (np.array(landmarks[eye_indices[1]]) + np.array(landmarks[eye_indices[2]])) / 2
    bottom = (np.array(landmarks[eye_indices[5]]) + np.array(landmarks[eye_indices[4]])) / 2
    return np.linalg.norm(top - bottom) / np.linalg.norm(left - right)

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
EAR_THRESHOLD = 0.2
CONSEC_FRAMES = 15
beep_cooldown = 10

# Streamlit UI
st.title("🎬 MIRINDA - AI Assistant with Live Attentiveness Monitor")
video_col, webcam_col = st.columns(2)

# Webcam feed
webcam_col.subheader("📷 Webcam Feed")
frame_display = webcam_col.empty()
timer_display = webcam_col.empty()
score_chart = webcam_col.empty()
run_webcam = webcam_col.checkbox("Start Webcam Monitor", key="run_webcam")

closed_eyes_frame_count = 0
last_beep_time = 0
attentiveness_scores = []
timestamps = []
cap = cv2.VideoCapture(0)

if run_webcam:
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)
        frame_h, frame_w = frame.shape[:2]
        attentive = True

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            mesh_points = np.array([[pt.x * frame_w, pt.y * frame_h] for pt in landmarks.landmark])
            left_ear = calculate_ear(mesh_points, LEFT_EYE)
            right_ear = calculate_ear(mesh_points, RIGHT_EYE)
            avg_ear = (left_ear + right_ear) / 2.0

            if avg_ear < EAR_THRESHOLD:
                closed_eyes_frame_count += 1
            else:
                closed_eyes_frame_count = 0

            if closed_eyes_frame_count >= CONSEC_FRAMES:
                attentive = False

            mp.solutions.drawing_utils.draw_landmarks(
                image=frame,
                landmark_list=landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_spec
            )

        # Status overlay
        status_text = "✅ Attentive" if attentive else "⚠️ Not Attentive"
        color = (0, 255, 0) if attentive else (0, 0, 255)
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        if not attentive and int(time.time()) % 2 == 0:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame_w, frame_h), (0, 0, 255), -1)
            alpha = 0.2
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        current_time = time.time()
        if not attentive and (current_time - last_beep_time > beep_cooldown):
            winsound.Beep(1000, 500)
            last_beep_time = current_time

        frame_display.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        elapsed_time = int(time.time() - start_time)
        attentiveness_scores.append(1 if attentive else 0)
        timestamps.append(elapsed_time)
        df = pd.DataFrame({"Attentiveness": attentiveness_scores}, index=[str(timedelta(seconds=t)) for t in timestamps])
        score_chart.line_chart(df)

        formatted_time = str(timedelta(seconds=elapsed_time))
        timer_display.markdown(f"⏱️ **Time Monitoring:** `{formatted_time}`")

        if not st.session_state.get("run_webcam", True):
            break

if not run_webcam and cap.isOpened():
    cap.release()

# Transcript splitting helper
def split_transcript(transcript, max_tokens=450):
    words = transcript.split()
    chunks = []
    chunk = []

    for word in words:
        chunk.append(word)
        if len(tokenizer(" ".join(chunk))["input_ids"]) > max_tokens:
            chunk.pop()
            chunks.append(" ".join(chunk))
            chunk = [word]
    if chunk:
        chunks.append(" ".join(chunk))
    return chunks

# Video transcript and Q&A
video_col.subheader("📁 Upload and Transcribe Video")
uploaded_file = video_col.file_uploader("Upload a video file (mp4 format)", type=["mp4"])

if uploaded_file:
    with open("video.mp4", "wb") as f:
        f.write(uploaded_file.read())
    video_col.video("video.mp4")

    with st.spinner("Transcribing with Whisper..."):
        result = whisper_model.transcribe("video.mp4")
        transcript = result["text"]

    video_col.subheader("📝 Transcript")
    video_col.text_area("Transcript Text", transcript, height=300)

    video_col.subheader("❓ Ask a Question")
    user_question = video_col.text_input("Type your question below:")

    if user_question:
        with st.spinner("Searching best answer..."):
            chunks = split_transcript(transcript)
            best_score = 0
            best_answer = "Sorry, I couldn't find a good answer."

            for chunk in chunks:
                try:
                    ans = qa_pipeline(question=user_question, context=chunk)
                    if ans['score'] > best_score:
                        best_score = ans['score']
                        best_answer = ans['answer']
                except:
                    continue

            video_col.success(f"Answer: {best_answer}")
