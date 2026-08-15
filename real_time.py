import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # Hide TF logs (0=all, 3=only errors)

import logging
logging.getLogger("absl").setLevel(logging.ERROR)  # Hide MediaPipe warnings
import warnings
warnings.filterwarnings("ignore")
import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

# MediaPipe Tasks API
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path="pose_landmarker_lite.task"
    ),
    running_mode=vision.RunningMode.VIDEO
)

pose = vision.PoseLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

history = deque(maxlen=10)
last_print_time = time.time()

def classify_action(landmarks):
    lm = landmarks[0]

    lw, rw = lm[15], lm[16]
    ls, rs = lm[11], lm[12]
    lk, rk = lm[25], lm[26]
    lh = lm[23]

    knee_diff = abs(lk.y - rk.y)
    hand_diff = abs(lw.x - rw.x)

    # HANDWAVING
    if lw.y < ls.y or rw.y < rs.y:
        if hand_diff > 0.1:
            return "handwaving"

    # HANDCLAPPING
    if hand_diff < 0.05 and lw.y < lh.y:
        return "handclapping"

    # BOXING
    if lw.y < lh.y and rw.y < lh.y:
        if hand_diff > 0.15:
            return "boxing"

    # RUNNING
    if knee_diff > 0.12:
        return "running"

    # JOGGING
    if 0.06 < knee_diff <= 0.12:
        return "jogging"

    # WALKING
    if 0.02 < knee_diff <= 0.06:
        return "walking"

    return "Uncertain"

frame_timestamp = 0
current_label = "Detecting..."

print("Starting Real-Time HAR...")
print("Press 'q' to quit\n")

while True:
    start_time = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = pose.detect_for_video(mp_image, frame_timestamp)
    frame_timestamp += 1

    if result.pose_landmarks:
        label = classify_action(result.pose_landmarks)

        history.append(label)
        current_label = max(set(history), key=history.count)

    # FPS calculation
    fps = 1.0 / (time.time() - start_time)

    # -----------------------------
    # PRINT TO CMD (every 1 sec)
    # -----------------------------
    if time.time() - last_print_time > 5:
        print(f"Prediction: {current_label} | FPS: {fps:.2f}")
        last_print_time = time.time()

    # -----------------------------
    # DISPLAY
    # -----------------------------
    cv2.putText(frame, f"Action: {current_label}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.imshow("Real-Time HAR", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()