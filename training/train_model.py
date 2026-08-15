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
    BatchNormalization,
    Bidirectional
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau
)

from preprocessing.dataset_builder import get_generators

# CONFIG

SEQUENCE_LENGTH = 20
IMG_SIZE = 160
NUM_CLASSES = 6
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.keras")

os.makedirs(MODEL_DIR, exist_ok=True)

train_gen, test_gen = get_generators()

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

# MODEL
model = Sequential([
    Input(shape=(SEQUENCE_LENGTH, IMG_SIZE, IMG_SIZE, 3)),

    TimeDistributed(base_model),
    TimeDistributed(GlobalAveragePooling2D()),
    TimeDistributed(BatchNormalization()),

    Bidirectional(LSTM(128, return_sequences=True)),
    Dropout(0.4),
    Bidirectional(LSTM(64)),
    Dropout(0.4),

    Dense(128, activation="relu"),
    Dropout(0.3),

    Dense(NUM_CLASSES, activation="softmax")
])

# COMPILE
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=["accuracy"]
)

# CALLBACKS
checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_accuracy",
    patience=10,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=4,
    verbose=1
)

# TRAIN

history = model.fit(
    train_gen,
    validation_data=test_gen,
    epochs=50,
    callbacks=[checkpoint, early_stop, reduce_lr]
)

print("Training completed.")


import matplotlib.pyplot as plt

# PLOT TRAINING CURVES
def plot_training(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    # Accuracy Plot
    plt.figure()
    plt.plot(epochs, acc)
    plt.plot(epochs, val_acc)
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['Train Accuracy', 'Validation Accuracy'])
    plt.grid(True)
    plt.show()

    # Loss Plot
    plt.figure()
    plt.plot(epochs, loss)
    plt.plot(epochs, val_loss)
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Train Loss', 'Validation Loss'])
    plt.grid(True)
    plt.show()

plot_training(history)