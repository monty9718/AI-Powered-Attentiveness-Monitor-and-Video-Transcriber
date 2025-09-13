# attentiveness_check.py
import cv2
import mediapipe as mp
import numpy as np
import time
import ctypes
import threading

def start_attentiveness_monitor():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
    drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1)

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

    closed_eyes_frame_count = 0
    popup_shown = False
    last_popup_time = 0
    popup_cooldown = 5  # seconds

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)
        frame_h, frame_w = frame.shape[:2]

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

            attentive = closed_eyes_frame_count < CONSEC_FRAMES
            current_time = time.time()

            if not attentive and (not popup_shown or current_time - last_popup_time > popup_cooldown):
                ctypes.windll.user32.MessageBoxW(0, "You are not attentive!", "⚠️ Attentiveness Alert", 0x40 | 0x1)
                popup_shown = True
                last_popup_time = current_time
            elif attentive:
                popup_shown = False

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
