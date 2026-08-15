import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import Sequence, to_categorical # type: ignore
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input # type: ignore

DATA_PATH = "data_frames"
BATCH_SIZE = 8
NUM_CLASSES = 6

actions = sorted(os.listdir(DATA_PATH))
label_map = {label: num for num, label in enumerate(actions)}

class DataGenerator(Sequence):

    def __init__(self, file_list, labels, batch_size=BATCH_SIZE, shuffle=True):
        self.file_list = file_list
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.file_list) / self.batch_size))

    def __getitem__(self, index):
        batch_files = self.file_list[index*self.batch_size:(index+1)*self.batch_size]
        batch_labels = self.labels[index*self.batch_size:(index+1)*self.batch_size]

        X, y = [], []

        for file_path, label in zip(batch_files, batch_labels):
            # data = np.load(file_path).astype("float32")
            # data = preprocess_input(data)
            data = np.load(file_path).astype("float32")

            # ===== Motion feature (NEW) =====
            motion = np.diff(data, axis=0)
            motion = np.concatenate([motion, motion[-1:]], axis=0)
            data = data + 0.5 * motion
            # ================================

            data = preprocess_input(data)

            # Simple augmentation
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

def get_generators():

    file_list = []
    labels = []

    for action in actions:
        action_path = os.path.join(DATA_PATH, action)

        for file in os.listdir(action_path):
            file_list.append(os.path.join(action_path, file))
            labels.append(label_map[action])

    train_files, test_files, train_labels, test_labels = train_test_split(
        file_list,
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=42
    )

    train_gen = DataGenerator(train_files, train_labels)
    test_gen = DataGenerator(test_files, test_labels, shuffle=False)

    return train_gen, test_gen
