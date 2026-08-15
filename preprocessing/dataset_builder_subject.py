import os
import numpy as np
import re
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

DATA_PATH = "data_frames"
BATCH_SIZE = 8
NUM_CLASSES = 6

actions = sorted(os.listdir(DATA_PATH))
label_map = {label: num for num, label in enumerate(actions)}

# -----------------------------
# Extract Subject ID
# -----------------------------
def extract_subject_id(filepath):
    filename = os.path.basename(filepath)
    match = re.search(r'person(\d+)', filename)
    if match:
        return int(match.group(1))
    return None


class DataGenerator(Sequence):

    def __init__(self, file_list, labels, batch_size=BATCH_SIZE, shuffle=True):
        self.file_list = file_list
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.samples = len(file_list)
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.file_list) / self.batch_size))

    def __getitem__(self, index):
        batch_files = self.file_list[index*self.batch_size:(index+1)*self.batch_size]
        batch_labels = self.labels[index*self.batch_size:(index+1)*self.batch_size]

        X, y = [], []

        for file_path, label in zip(batch_files, batch_labels):
            data = np.load(file_path).astype("float32")

            # Motion enhancement
            motion = np.diff(data, axis=0)
            motion = np.concatenate([motion, motion[-1:]], axis=0)
            data = data + 0.5 * motion

            data = preprocess_input(data)

            if self.shuffle and np.random.rand() < 0.5:
                data = np.flip(data, axis=2)

            X.append(data)
            y.append(label)

        return np.array(X), to_categorical(y, NUM_CLASSES)

    def on_epoch_end(self):
        if self.shuffle:
            temp = list(zip(self.file_list, self.labels))
            np.random.shuffle(temp)
            self.file_list, self.labels = zip(*temp)


def get_generators_subject_split():

    subjects = {}

    for action in actions:
        action_path = os.path.join(DATA_PATH, action)

        for file in os.listdir(action_path):
            full_path = os.path.join(action_path, file)
            subject_id = extract_subject_id(full_path)

            if subject_id not in subjects:
                subjects[subject_id] = []

            subjects[subject_id].append((full_path, label_map[action]))

    subject_ids = sorted(subjects.keys())

    # 16 train subjects, 9 test subjects
    train_ids = subject_ids[:16]
    test_ids = subject_ids[16:]

    train_files, train_labels = [], []
    test_files, test_labels = [], []

    for sid in train_ids:
        for path, label in subjects[sid]:
            train_files.append(path)
            train_labels.append(label)

    for sid in test_ids:
        for path, label in subjects[sid]:
            test_files.append(path)
            test_labels.append(label)

    train_gen = DataGenerator(train_files, train_labels)
    test_gen = DataGenerator(test_files, test_labels, shuffle=False)

    return train_gen, test_gen