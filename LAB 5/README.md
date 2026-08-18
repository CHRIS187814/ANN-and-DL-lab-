# Lab 5 — CNN for Binary Image Classification (Cats vs Dogs)

TensorFlow / Keras implementation of a Convolutional Neural Network that
classifies images into two classes, trained with data augmentation and
evaluated with a full per-epoch error log, confusion matrix, and sample
predictions.

## Contents

| File | Purpose |
|---|---|
| `lab5_cnn_cats_vs_dogs.py` | Main script — CNN on the **real Kaggle Cats-vs-Dogs** dataset (or any `train/`, `val/` folder of two classes). Use this to run the lab on your own machine with the real dataset. |
| `make_dataset.py` | Generates a synthetic two-class image dataset (cat-like / dog-like cartoon faces) — a stand-in used only because this sandbox has no internet access to Kaggle. Produces `data/train/{cats,dogs}` and `data/val/{cats,dogs}`. |
| `train.py` | The version actually run in this sandbox against the synthetic dataset. Same architecture and training loop as the main script, plus the code that produced every PNG and number in the report. |
| `results.json` | Raw numeric results of the run: per-epoch loss/accuracy, final metrics, confusion matrix, classification report. |
| `training_curves.png` | Error (loss) and accuracy vs. epoch, train and validation. |
| `confusion_matrix.png` | Confusion matrix on the validation set. |
| `sample_predictions.png` | A grid of validation images with true/predicted labels and confidence. |
| `sample_images.png` | Sample of the generated training images (cats vs dogs). |
| `Lab5_CNN_Report.docx` | Full written report: objectives, methodology, architecture, per-epoch error table, results, observations, and conclusions — with all PNGs embedded. |
| `cats_vs_dogs_cnn.keras` | The trained model, saved in Keras format. |

## Why a synthetic dataset?

The standard lab dataset (Kaggle "Dogs vs Cats") requires downloading from
`kaggle.com`, and TensorFlow's `cats_vs_dogs` dataset via
`tensorflow_datasets` requires downloading from Google Cloud Storage.
Neither is reachable from this sandboxed environment's network allow-list.
To still produce **real, non-fabricated** training curves, confusion
matrices, and metrics — rather than invented numbers — `make_dataset.py`
procedurally draws simple cat-like and dog-like faces (distinguished by
ear shape, eye shape, and snout) with randomized position, rotation, color,
and pixel noise, and the full CNN pipeline was actually trained and
evaluated on them.

**The architecture, preprocessing, augmentation, training loop, and
evaluation code are 100% identical to what you'd use on the real
dataset** — only the image source differs. `lab5_cnn_cats_vs_dogs.py` is
the ready-to-run version for the real Kaggle dataset.

## How to run this on the real Cats vs Dogs dataset

1. Download the dataset from
   [Kaggle: Dogs vs Cats](https://www.kaggle.com/c/dogs-vs-cats/data) and
   unzip it.
2. Arrange the images into this folder structure (an 80/20 split is a
   reasonable default):
   ```
   data/train/cats/*.jpg
   data/train/dogs/*.jpg
   data/val/cats/*.jpg
   data/val/dogs/*.jpg
   ```
3. Install dependencies:
   ```bash
   pip install tensorflow matplotlib scikit-learn --break-system-packages
   ```
4. Run:
   ```bash
   python3 lab5_cnn_cats_vs_dogs.py
   ```
   This prints the per-epoch error/accuracy table to the console, saves
   `training_curves.png`, evaluates on the validation set, prints a
   confusion matrix and classification report, and saves the trained
   model as `cats_vs_dogs_cnn.keras`.

## How to reproduce the exact run in this repo (synthetic dataset)

```bash
pip install tensorflow-cpu matplotlib scikit-learn pillow --break-system-packages
python3 make_dataset.py   # generates data/train and data/val
python3 train.py          # trains the CNN, saves all PNGs + results.json
```

## Model architecture

```
Input (64×64×3)
 → Conv2D(32, 3×3, ReLU) → MaxPooling2D(2×2)
 → Conv2D(64, 3×3, ReLU) → MaxPooling2D(2×2)
 → Conv2D(128, 3×3, ReLU) → MaxPooling2D(2×2)
 → Flatten → Dropout(0.5)
 → Dense(128, ReLU)
 → Dense(1, Sigmoid)
```
Total parameters: 683,329. Loss: binary cross-entropy. Optimizer: Adam
(lr = 1e-3). Batch size: 16. Epochs: 15.

## How training minimises the error (feedforward + backpropagation)

Each training step, repeated automatically inside `model.fit()`:

1. **Feedforward** — a batch of images passes through every conv/pool/
   dense layer, producing a predicted probability `p = P(dog)` for each
   image.
2. **Error (loss)** — binary cross-entropy is computed:
   `L = -(y·log(p) + (1-y)·log(1-p))`, averaged over the batch. This is
   the number reported as "loss" every epoch.
3. **Backpropagation** — the gradient of `L` with respect to every
   weight (`∂L/∂w`) is computed via the chain rule, propagated layer by
   layer from the output back to the first convolutional layer.
4. **Weight update** — Adam updates every weight,
   `w := w − lr·∂L/∂w` (with per-parameter adaptive learning rates), so
   the error is smaller on the next forward pass.

The per-epoch loss values in `results.json` / the report's error table
are the direct, measured record of this optimisation process.

## Results summary (synthetic dataset run)

- Final validation accuracy: **100%**
- Final validation loss: **~8.7e-05**
- Confusion matrix: 60/60 cats and 60/60 dogs correctly classified
  (zero false positives, zero false negatives)
- Training loss dropped from 0.47 (epoch 1) to near-zero by epoch 4–5,
  with occasional single-epoch spikes caused by aggressive data
  augmentation on small batches — expected behaviour, not a training
  failure (see `Lab5_CNN_Report.docx`, Section 7, for full discussion).

See `Lab5_CNN_Report.docx` for the full write-up including all figures,
the per-epoch error table, and the Observations/Conclusion sections
mapped to each learning outcome.

## Requirements

```
tensorflow (or tensorflow-cpu) >= 2.15
matplotlib
scikit-learn
pillow      (only needed for make_dataset.py)
```
