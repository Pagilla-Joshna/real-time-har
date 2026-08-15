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
    Bidirectional,
    Attention
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau
)

from preprocessing.dataset_builder_subject import get_generators_subject_split

# REPRODUCIBILITY

tf.random.set_seed(42)

# CONFIG
SEQUENCE_LENGTH = 20
IMG_SIZE = 160
NUM_CLASSES = 6

MODEL_DIR = "models"
RESULTS_DIR = "Results/attention"

MODEL_PATH = os.path.join(MODEL_DIR, "best_model_attention.keras")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

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

# MODEL (Attention)

inputs = Input(shape=(SEQUENCE_LENGTH, IMG_SIZE, IMG_SIZE, 3))

x = TimeDistributed(base_model)(inputs)
x = TimeDistributed(GlobalAveragePooling2D())(x)
x = TimeDistributed(BatchNormalization())(x)

x = Bidirectional(LSTM(128, return_sequences=True))(x)
x = Dropout(0.4)(x)

attention = Attention()([x, x])

x = Bidirectional(LSTM(64))(attention)
x = Dropout(0.4)(x)

x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)

outputs = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(3e-4),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=["accuracy"]
)

callbacks = [
    ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1),
    EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, verbose=1)
]

# TRAIN
history = model.fit(
    train_gen,
    validation_data=test_gen,
    epochs=40,
    callbacks=callbacks
)

# SAVE FINAL ACCURACY
best_acc = max(history.history["val_accuracy"])

with open(os.path.join(RESULTS_DIR, "accuracy.txt"), "w") as f:
    f.write(f"Best Validation Accuracy: {best_acc:.4f}")

print(f"Best Validation Accuracy: {best_acc:.4f}")

# PLOT ACCURACY CURVE
plt.figure()
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["Train", "Validation"])
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "accuracy_curve.png"))
plt.close()

# PLOT LOSS CURVE
plt.figure()
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(["Train", "Validation"])
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "loss_curve.png"))
plt.close()

print("Training completed. Curves saved in Results/attention/")