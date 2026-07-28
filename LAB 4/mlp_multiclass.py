"""
Keras MLP for Multiclass Classification
-----------------------------------------
Dataset : sklearn Wine dataset (13 numeric features, 3 classes of wine cultivar)
Model   : Sequential MLP (Dense -> Dense -> Softmax output)
Author  : Mario  |  ElevateEd / BSc Data Science & AI coursework

This script trains a small feed-forward neural network to classify wine
samples into one of three cultivars based on 13 chemical measurements.
Run: python mlp_multiclass.py
Outputs: metrics.json, training_history.png, confusion_matrix.png, classification_report.txt
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 1. Reproducibility -----------------------------------------------------
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# 2. Load data -------------------------------------------------------------
data = load_wine()
X, y = data.data, data.target
feature_names = data.feature_names
class_names = data.target_names
num_classes = len(class_names)

print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}, Classes: {num_classes}")

# 3. Train / validation / test split ---------------------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=SEED, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp
)

# 4. Feature scaling (fit ONLY on training data) -----------------------------
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

# 5. One-hot encode labels for categorical_crossentropy ----------------------
y_train_cat = keras.utils.to_categorical(y_train, num_classes)
y_val_cat = keras.utils.to_categorical(y_val, num_classes)
y_test_cat = keras.utils.to_categorical(y_test, num_classes)

# 6. Build the MLP -----------------------------------------------------------
model = keras.Sequential([
    layers.Input(shape=(X_train_s.shape[1],)),
    layers.Dense(32, activation="relu", name="hidden_1"),
    layers.Dropout(0.2),
    layers.Dense(16, activation="relu", name="hidden_2"),
    layers.Dense(num_classes, activation="softmax", name="output"),
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# 7. Train --------------------------------------------------------------------
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=15, restore_best_weights=True
)

history = model.fit(
    X_train_s, y_train_cat,
    validation_data=(X_val_s, y_val_cat),
    epochs=150,
    batch_size=8,
    callbacks=[early_stop],
    verbose=2,
)

# 8. Evaluate on held-out test set --------------------------------------------
test_loss, test_acc = model.evaluate(X_test_s, y_test_cat, verbose=0)
y_pred_prob = model.predict(X_test_s, verbose=0)
y_pred = np.argmax(y_pred_prob, axis=1)

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=class_names, digits=3)

print("\n=== TEST RESULTS ===")
print(f"Test Loss: {test_loss:.4f}  |  Test Accuracy: {test_acc:.4f}")
print(report)

# 9. Save classification report -----------------------------------------------
with open("classification_report.txt", "w") as f:
    f.write(f"Test Loss: {test_loss:.4f}\nTest Accuracy: {test_acc:.4f}\n\n")
    f.write(report)

# 10. Save metrics.json (also feeds the README results table) -----------------
epochs_ran = len(history.history["loss"])
metrics = {
    "epochs_ran": epochs_ran,
    "final_train_accuracy": float(history.history["accuracy"][-1]),
    "final_val_accuracy": float(history.history["val_accuracy"][-1]),
    "final_train_loss": float(history.history["loss"][-1]),
    "final_val_loss": float(history.history["val_loss"][-1]),
    "test_loss": float(test_loss),
    "test_accuracy": float(test_acc),
    "confusion_matrix": cm.tolist(),
    "class_names": class_names.tolist(),
}
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# 11. Plot training curves -----------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(history.history["loss"], label="train loss")
axes[0].plot(history.history["val_loss"], label="val loss")
axes[0].set_title("Loss over Epochs")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Categorical Crossentropy")
axes[0].legend()

axes[1].plot(history.history["accuracy"], label="train accuracy")
axes[1].plot(history.history["val_accuracy"], label="val accuracy")
axes[1].set_title("Accuracy over Epochs")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()

plt.tight_layout()
plt.savefig("training_history.png", dpi=150)
plt.close()

# 12. Plot confusion matrix ------------------------------------------------------
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix - Test Set")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()

print("\nSaved: metrics.json, training_history.png, confusion_matrix.png, classification_report.txt")
