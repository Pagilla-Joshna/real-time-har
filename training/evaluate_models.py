import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tensorflow as tf

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize

from preprocessing.dataset_builder_subject import get_generators_subject_split

# CONFIG

MODELS = {
    "cnn": "models/best_model_cnn.keras",
    "cnn_lstm": "models/best_model_cnn_lstm.keras",
    "subject" : "models/best_model_subject.keras",
    "random" : "models/best_model.keras",
    "attention": "models/best_model_attention.keras"
}

NUM_CLASSES = 6
BASE_RESULTS_DIR = "Results"

# LOAD SUBJECT-WISE DATA

print("Loading subject-wise test data...")
train_gen, test_gen = get_generators_subject_split()

actions = sorted(os.listdir("data_frames"))
print("Classes:", actions)

# EVALUATE FUNCTION

def evaluate_model(model_name, model_path):

    print(f"\nEvaluating {model_name.upper()} model...")
    
    # Create model-specific results folder
    results_dir = os.path.join(BASE_RESULTS_DIR, model_name)
    os.makedirs(results_dir, exist_ok=True)

    # Load model
    model = tf.keras.models.load_model(model_path)

    y_true = []
    y_pred = []
    y_prob = []

    for i in range(len(test_gen)):
        X, y = test_gen[i]
        preds = model.predict(X, verbose=0)

        y_true.extend(np.argmax(y, axis=1))
        y_pred.extend(np.argmax(preds, axis=1))
        y_prob.extend(preds)

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    y_prob_arr = np.array(y_prob)

    # ================= ACCURACY =================
    acc = accuracy_score(y_true_arr, y_pred_arr)
    print(f"{model_name} Accuracy: {acc:.4f}")

    with open(os.path.join(results_dir, "accuracy.txt"), "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")

    # ================= CONFUSION MATRIX =================
    cm = confusion_matrix(y_true_arr, y_pred_arr)

    plt.figure(figsize=(7, 6))
    sns.heatmap(cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=actions,
                yticklabels=actions)

    plt.title(f"{model_name.upper()} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))
    plt.close()

    # ================= CLASSIFICATION REPORT =================
    report = classification_report(y_true_arr, y_pred_arr, target_names=actions)

    with open(os.path.join(results_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    print(report)

    # ================= ROC CURVE =================
    y_true_bin = label_binarize(y_true_arr, classes=list(range(NUM_CLASSES)))

    plt.figure(figsize=(8, 6))

    for i in range(NUM_CLASSES):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob_arr[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{actions[i]} (AUC={roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_name.upper()} ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "roc_curve.png"))
    plt.close()

    print(f"{model_name} results saved in {results_dir}")

# RUN FOR ALL MODELS

for name, path in MODELS.items():
    if os.path.exists(path):
        evaluate_model(name, path)
    else:
        print(f"Model not found: {path}")

print("\nAll evaluations completed.")