import cv2
import os
import numpy as np
from tqdm import tqdm

DATASET_PATH = "data"
OUTPUT_PATH = "data_frames"
# IMG_SIZE = 160
# SEQUENCE_LENGTH = 16
SEQUENCE_LENGTH = 20
IMG_SIZE = 160

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

    while len(frames) < SEQUENCE_LENGTH:
        frames.append(frames[-1])

    return np.array(frames)

def process_dataset():
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    for action in os.listdir(DATASET_PATH):
        action_path = os.path.join(DATASET_PATH, action)

        if not os.path.isdir(action_path):
            continue

        save_path = os.path.join(OUTPUT_PATH, action)
        os.makedirs(save_path, exist_ok=True)

        for video in tqdm(os.listdir(action_path), desc=action):
            if not video.endswith(".avi"):
                continue

            video_path = os.path.join(action_path, video)
            frames = extract_frames(video_path)

            np.save(os.path.join(save_path, video.split(".")[0]), frames)

if __name__ == "__main__":
    process_dataset()
