"""
Lab 5: CNN for Binary Image Classification (Cats vs Dogs)
TensorFlow / Keras implementation

Covers:
  - CNN architecture (Conv2D, Pooling, Dense layers)
  - Image preprocessing + data augmentation
  - Training with feedforward + backpropagation (handled internally by
    model.fit via the optimizer, but instrumented here so you can see the
    per-epoch error/loss explicitly)
  - Evaluation, confusion matrix, and sample predictions
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers
from sklearn.metrics import confusion_matrix, classification_report

tf.random.set_seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# 1. DATASET
# ---------------------------------------------------------------------------
# Option A: Kaggle "Dogs vs Cats" — download from
#   https://www.kaggle.com/c/dogs-vs-cats/data
# and unzip so you have:
#   data/train/cats/*.jpg
#   data/train/dogs/*.jpg
#   data/val/cats/*.jpg
#   data/val/dogs/*.jpg
#
# Option B (no manual download): use tensorflow_datasets, which gives the
# same Cats-vs-Dogs dataset directly. Uncomment the block below if you'd
# rather use that instead of a local folder.
#
# import tensorflow_datasets as tfds
# (raw_train, raw_val), info = tfds.load(
#     "cats_vs_dogs",
#     split=["train[:80%]", "train[80%:]"],
#     as_supervised=True,
#     with_info=True,
# )

TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
IMG_SIZE = (150, 150)
BATCH_SIZE = 32

# ---------------------------------------------------------------------------
# 2. PREPROCESSING + DATA AUGMENTATION
# ---------------------------------------------------------------------------
# Augmentation is applied ONLY to training data — it artificially expands
# the dataset (rotations, shifts, flips, zoom) so the model generalises
# instead of memorising the training images.
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,          # normalise pixel values to [0, 1]
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

# Validation data is only rescaled — never augmented, so evaluation reflects
# real, unmodified images.
val_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",   # 2 classes -> single sigmoid output (0 = cat, 1 = dog)
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
)

# ---------------------------------------------------------------------------
# 3. CNN ARCHITECTURE
# ---------------------------------------------------------------------------
# Each Conv2D + MaxPooling2D block extracts progressively higher-level
# features (edges -> textures -> shapes -> object parts). Flatten + Dense
# layers then combine those features into a single binary decision.
model = models.Sequential([
    layers.Input(shape=(150, 150, 3)),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dropout(0.5),          # reduces overfitting
    layers.Dense(512, activation="relu"),
    layers.Dense(1, activation="sigmoid"),   # binary output: P(dog)
])

model.summary()

# ---------------------------------------------------------------------------
# 4. COMPILE — loss function + optimizer (this is where backprop is wired up)
# ---------------------------------------------------------------------------
# Loss:      binary_crossentropy — the "error" being minimised each epoch:
#               L = -( y*log(p) + (1-y)*log(1-p) )
# Optimizer: Adam — computes the gradient of L w.r.t. every weight via
#            backpropagation (chain rule through every layer) and updates
#            weights: w := w - lr * dL/dw
model.compile(
    loss="binary_crossentropy",
    optimizer=optimizers.Adam(learning_rate=1e-4),
    metrics=["accuracy"],
)

# ---------------------------------------------------------------------------
# 5. TRAIN — feedforward + backpropagation happen inside model.fit()
# ---------------------------------------------------------------------------
# What happens on every batch, every epoch:
#   1. FEEDFORWARD : input images pass through conv/pool/dense layers to
#      produce a predicted probability p.
#   2. ERROR       : binary_crossentropy(y_true, p) is computed -> this is
#      the "error value" for that batch.
#   3. BACKPROP    : gradients of the error w.r.t. every weight are computed
#      via the chain rule, layer by layer, from output back to input.
#   4. UPDATE      : Adam uses those gradients to adjust every weight so the
#      error decreases on the next forward pass.
# Keras does steps 1-4 automatically inside .fit(); the History object
# records the resulting per-epoch error (loss) and accuracy.
EPOCHS = 20

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=val_generator,
    validation_steps=val_generator.samples // BATCH_SIZE,
)

# ---------------------------------------------------------------------------
# 6. PER-EPOCH ERROR TABLE
# ---------------------------------------------------------------------------
print("\nEpoch | Train Loss (error) | Train Acc | Val Loss (error) | Val Acc")
print("-" * 70)
for epoch in range(EPOCHS):
    tr_loss = history.history["loss"][epoch]
    tr_acc = history.history["accuracy"][epoch]
    val_loss = history.history["val_loss"][epoch]
    val_acc = history.history["val_accuracy"][epoch]
    print(f"{epoch+1:>5} | {tr_loss:>19.4f} | {tr_acc:>9.4f} | "
          f"{val_loss:>17.4f} | {val_acc:>7.4f}")

# ---------------------------------------------------------------------------
# 7. PLOT ERROR (LOSS) AND ACCURACY CURVES
# ---------------------------------------------------------------------------
epochs_range = range(1, EPOCHS + 1)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, history.history["loss"], label="Train Error (loss)")
plt.plot(epochs_range, history.history["val_loss"], label="Val Error (loss)")
plt.xlabel("Epoch")
plt.ylabel("Binary Cross-Entropy Error")
plt.title("Error vs Epoch (feedforward + backprop minimising this)")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, history.history["accuracy"], label="Train Accuracy")
plt.plot(epochs_range, history.history["val_accuracy"], label="Val Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Epoch")
plt.legend()

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.show()

# ---------------------------------------------------------------------------
# 8. EVALUATE ON VALIDATION SET
# ---------------------------------------------------------------------------
val_loss, val_acc = model.evaluate(val_generator)
print(f"\nFinal Validation Error (loss): {val_loss:.4f}")
print(f"Final Validation Accuracy:     {val_acc:.4f}")

# ---------------------------------------------------------------------------
# 9. CONFUSION MATRIX + CLASSIFICATION REPORT
# ---------------------------------------------------------------------------
val_generator.reset()
y_true = val_generator.classes
y_pred_prob = model.predict(val_generator, steps=val_generator.samples // BATCH_SIZE + 1)
y_pred = (y_pred_prob[: len(y_true)] > 0.5).astype(int).ravel()

cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=list(train_generator.class_indices.keys())))

# ---------------------------------------------------------------------------
# 10. SAVE THE TRAINED MODEL
# ---------------------------------------------------------------------------
model.save("cats_vs_dogs_cnn.keras")
print("\nModel saved to cats_vs_dogs_cnn.keras")
