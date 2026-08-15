import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    TimeDistributed,
    LSTM,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    BatchNormalization
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau
)

from preprocessing.dataset_builder_subject import get_generators_subject_split

# CONFIG
SEQUENCE_LENGTH = 20
IMG_SIZE = 160
NUM_CLASSES = 6

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model_cnn_lstm.keras")
os.makedirs(MODEL_DIR, exist_ok=True)

train_gen, test_gen = get_generators_subject_split()

# BACKBONE
base_model = MobileNetV2(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    alpha=0.75
)

base_model.trainable = True
for layer in base_model.layers[:-50]:
    layer.trainable = False

# MODEL (CNN + LSTM)
model = Sequential([
    Input(shape=(SEQUENCE_LENGTH, IMG_SIZE, IMG_SIZE, 3)),

    TimeDistributed(base_model),
    TimeDistributed(GlobalAveragePooling2D()),
    TimeDistributed(BatchNormalization()),

    LSTM(128, return_sequences=False),
    Dropout(0.4),

    Dense(128, activation="relu"),
    Dropout(0.3),

    Dense(NUM_CLASSES, activation="softmax")
])

# COMPILE
model.compile(
    optimizer=tf.keras.optimizers.Adam(3e-4),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=["accuracy"]
)

# CALLBACKS
callbacks = [
    ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True),
    EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3)
]

# TRAIN
history = model.fit(
    train_gen,
    validation_data=test_gen,
    epochs=25,
    callbacks=callbacks
)

print("CNN + LSTM training completed.")