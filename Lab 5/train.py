import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers
from sklearn.metrics import confusion_matrix, classification_report

tf.random.set_seed(42)
np.random.seed(42)

IMG_SIZE = (64, 64)
BATCH_SIZE = 16
EPOCHS = 15

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest",
)
val_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_gen = train_datagen.flow_from_directory(
    "data/train", target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="binary", shuffle=True, seed=42)
val_gen = val_datagen.flow_from_directory(
    "data/val", target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="binary", shuffle=False)

print("Class indices:", train_gen.class_indices)

model = models.Sequential([
    layers.Input(shape=(64, 64, 3)),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(128, activation="relu"),
    layers.Dense(1, activation="sigmoid"),
])
model.compile(loss="binary_crossentropy", optimizer=optimizers.Adam(learning_rate=1e-3), metrics=["accuracy"])
model.summary()

history = model.fit(
    train_gen,
    steps_per_epoch=train_gen.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=val_gen,
    validation_steps=val_gen.samples // BATCH_SIZE,
)

# ---- per-epoch error table ----
epoch_rows = []
for e in range(EPOCHS):
    epoch_rows.append({
        "epoch": e + 1,
        "train_loss": float(history.history["loss"][e]),
        "train_acc": float(history.history["accuracy"][e]),
        "val_loss": float(history.history["val_loss"][e]),
        "val_acc": float(history.history["val_accuracy"][e]),
    })
    print(f"Epoch {e+1:2d} | train_loss={epoch_rows[-1]['train_loss']:.4f} "
          f"train_acc={epoch_rows[-1]['train_acc']:.4f} | "
          f"val_loss={epoch_rows[-1]['val_loss']:.4f} val_acc={epoch_rows[-1]['val_acc']:.4f}")

# ---- training curves ----
epochs_range = range(1, EPOCHS + 1)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, history.history["loss"], marker="o", label="Train Error (loss)")
plt.plot(epochs_range, history.history["val_loss"], marker="o", label="Val Error (loss)")
plt.xlabel("Epoch"); plt.ylabel("Binary Cross-Entropy Error"); plt.title("Error vs Epoch")
plt.legend(); plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(epochs_range, history.history["accuracy"], marker="o", label="Train Accuracy")
plt.plot(epochs_range, history.history["val_accuracy"], marker="o", label="Val Accuracy")
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Accuracy vs Epoch")
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.close()

# ---- final evaluation ----
val_loss, val_acc = model.evaluate(val_gen)
print(f"Final val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

val_gen.reset()
y_true = val_gen.classes
y_prob = model.predict(val_gen, steps=int(np.ceil(val_gen.samples / BATCH_SIZE)))
y_pred = (y_prob[: len(y_true)] > 0.5).astype(int).ravel()

class_names = list(train_gen.class_indices.keys())
cm = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
print(cm)
print(classification_report(y_true, y_pred, target_names=class_names))

# ---- confusion matrix plot ----
plt.figure(figsize=(5, 4.5))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix (Validation)")
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                  color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.xlabel("Predicted"); plt.ylabel("True")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()

# ---- sample predictions grid ----
val_gen.reset()
imgs, labels = next(val_gen)
probs = model.predict(imgs)
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(imgs[i])
    true_lbl = class_names[int(labels[i])]
    pred_lbl = class_names[int(probs[i][0] > 0.5)]
    correct = true_lbl == pred_lbl
    ax.set_title(f"true={true_lbl}\npred={pred_lbl} ({probs[i][0]:.2f})",
                 color="green" if correct else "red", fontsize=9)
    ax.axis("off")
plt.suptitle("Sample Validation Predictions")
plt.tight_layout()
plt.savefig("sample_predictions.png", dpi=150)
plt.close()

model.save("cats_vs_dogs_cnn.keras")

with open("results.json", "w") as f:
    json.dump({
        "epoch_rows": epoch_rows,
        "final_val_loss": float(val_loss),
        "final_val_acc": float(val_acc),
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
        "classification_report": report,
        "total_params": int(model.count_params()),
    }, f, indent=2)

print("DONE")
