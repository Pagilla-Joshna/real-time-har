import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# =========================
# CONFIG
# =========================
VIDEO_PATH = "test/test_vid_2.avi"
SEQUENCE_LENGTH = 20
IMG_SIZE = 160

MODEL_PATH = "models/best_model_subject.keras"

# IMPORTANT: match training order exactly
actions = sorted(["boxing", "handclapping", "handwaving",
                  "jogging", "running", "walking"])

# =========================
# LOAD MODEL
# =========================
model = tf.keras.models.load_model(MODEL_PATH)

# =========================
# EXTRACT FRAMES (MATCH TRAINING)
# =========================
def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip = max(total_frames // SEQUENCE_LENGTH, 1)

    for i in range(SEQUENCE_LENGTH):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * skip)
        ret, frame = cap.read()

        if not ret:
            continue

        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    # pad if needed
    while len(frames) < SEQUENCE_LENGTH:
        frames.append(frames[-1])

    return np.array(frames, dtype="float32")

# =========================
# PREPROCESS
# =========================
frames = extract_frames(VIDEO_PATH)

# Motion feature (same as training)
motion = np.diff(frames, axis=0)
motion = np.concatenate([motion, motion[-1:]], axis=0)
frames = frames + 0.5 * motion

# MobileNet preprocessing
frames = preprocess_input(frames)

input_data = np.expand_dims(frames, axis=0)

# =========================
# PREDICTION
# =========================
preds = model.predict(input_data, verbose=0)[0]

print("\n========== RESULT ==========")
for i, p in enumerate(preds):
    print(f"{actions[i]}: {p:.4f}")

class_id = np.argmax(preds)
confidence = preds[class_id]

print("\nPredicted Action:", actions[class_id])
print("Confidence:", round(float(confidence), 4))
print("============================\n")