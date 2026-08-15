import os
import tensorflow as tf
import matplotlib.pyplot as plt

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

from preprocessing.dataset_builder_subject import get_generators_subject_split


SEQUENCE_LENGTH = 20
IMG_SIZE = 160
NUM_CLASSES = 6

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model_subject.keras")

RESULTS_DIR = os.path.join("Results", "subject")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# Load dataset generators
train_gen, test_gen = get_generators_subject_split()


# Base CNN for spatial features
base_model = MobileNetV2(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    alpha=0.75
)

# Fine-tuning
base_model.trainable = True
for layer in base_model.layers[:-50]:
    layer.trainable = False


# Model Architecture
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


# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=["accuracy"]
)


# Callbacks
callbacks = [

    ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),

    EarlyStopping(
        monitor="val_accuracy",
        patience=10,
        restore_best_weights=True
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=4,
        verbose=1
    )
]


# Train model
history = model.fit(
    train_gen,
    validation_data=test_gen,
    epochs=50,
    callbacks=callbacks
)

# Plot Accuracy Curve


plt.figure()

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()

accuracy_path = os.path.join(RESULTS_DIR, "accuracy_curve.png")

plt.savefig(accuracy_path, dpi=300)
plt.close()

# Plot Loss Curve

plt.figure()

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()

loss_path = os.path.join(RESULTS_DIR, "loss_curve.png")

plt.savefig(loss_path, dpi=300)
plt.close()


print("Training completed.")
print(f"Accuracy curve saved to: {accuracy_path}")
print(f"Loss curve saved to: {loss_path}")