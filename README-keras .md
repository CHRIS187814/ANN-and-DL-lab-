# Keras MLP for Multiclass Classification

**Dataset:** Wine (`sklearn.datasets.load_wine`) — 178 samples, 13 chemical features, 3 cultivar classes
**Model:** Sequential MLP — `Dense(32, relu) → Dropout(0.2) → Dense(16, relu) → Dense(3, softmax)`
**File:** `mlp_multiclass.py`

---

## 1. What this program does

1. Loads the Wine dataset and splits it 70% train / 15% validation / 15% test (stratified, so each split keeps the same class proportions).
2. Standardizes features (mean 0, std 1) using a scaler fit **only** on the training set.
3. One-hot encodes the 3 class labels.
4. Builds and trains a small feed-forward neural network with `categorical_crossentropy` loss and the Adam optimizer.
5. Uses `EarlyStopping` to halt training once validation loss stops improving, and restores the best-performing weights.
6. Evaluates on a held-out test set and saves the confusion matrix, classification report, and training curves.

---

## 2. Glossary — uncommon terms used in the program

| Term | Meaning |
|---|---|
| **MLP (Multi-Layer Perceptron)** | A feed-forward neural network with one or more hidden layers between input and output — the simplest "deep" architecture. |
| **Softmax** | The output-layer activation for multiclass problems. Converts raw scores (logits) into probabilities that sum to 1 across all classes. |
| **Categorical Crossentropy** | The loss function for multiclass classification with one-hot labels. It penalizes the model more heavily the more confidently wrong it is. |
| **One-hot encoding** | Represents a class label as a vector of 0s with a single 1 in the position of the correct class (e.g. class 1 of 3 → `[0, 1, 0]`). Required because `categorical_crossentropy` compares probability vectors, not raw integers. |
| **ReLU (Rectified Linear Unit)** | Hidden-layer activation function: `f(x) = max(0, x)`. Cheap to compute and helps avoid vanishing gradients compared to sigmoid/tanh. |
| **Dropout** | A regularization layer that randomly "switches off" a fraction of neurons (here, 20%) during each training step, forcing the network not to over-rely on any single neuron. Active only during training. |
| **Epoch** | One full pass of the model over the entire training dataset. |
| **Batch size** | Number of samples processed before the model's weights are updated once. Smaller batches → noisier but more frequent updates. |
| **Adam optimizer** | An adaptive learning-rate optimization algorithm; combines ideas from momentum and RMSProp to converge faster than plain gradient descent. |
| **Learning rate** | Step size for weight updates. Too high → training oscillates/diverges; too low → training is slow or gets stuck. |
| **Validation split** | A slice of data set aside (not used for weight updates) purely to monitor how well the model generalizes during training. |
| **Early stopping** | A callback that stops training once a monitored metric (here, validation loss) stops improving for a set number of epochs (`patience`), preventing wasted computation and overfitting. |
| **Overfitting** | When a model performs very well on training data but progressively worse on unseen (validation/test) data — a sign it has "memorized" rather than "learned." |
| **Confusion matrix** | A table showing actual vs. predicted classes, revealing exactly which classes get confused with which. |
| **Precision** | Of all samples the model *predicted* as class X, what fraction actually were class X. |
| **Recall** | Of all samples that *actually are* class X, what fraction the model correctly found. |
| **F1-score** | Harmonic mean of precision and recall — a single number balancing both. |
| **Stratified split** | A train/test split that preserves the original proportion of each class in every subset. |

---

## 3. Results

### 3.1 Model architecture

| Layer | Type | Output Shape | Parameters |
|---|---|---|---|
| hidden_1 | Dense (ReLU) | (None, 32) | 448 |
| dropout | Dropout (0.2) | (None, 32) | 0 |
| hidden_2 | Dense (ReLU) | (None, 16) | 528 |
| output | Dense (Softmax) | (None, 3) | 51 |
| **Total** | | | **1,027** |

### 3.2 Training summary

| Metric | Value |
|---|---|
| Epochs actually run (early stopping) | 18 / 150 |
| Final training accuracy | 1.000 |
| Final training loss | 0.0030 |
| Final validation accuracy | 0.963 |
| Final validation loss (at restored best epoch) | 0.099 |

### 3.3 Test set performance

| Metric | Value |
|---|---|
| Test loss | 0.105 |
| Test accuracy | 0.926 (25/27 correct) |

### 3.4 Per-class report (test set, 27 samples)

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| class_0 | 0.900 | 1.000 | 0.947 | 9 |
| class_1 | 0.909 | 0.909 | 0.909 | 11 |
| class_2 | 1.000 | 0.857 | 0.923 | 7 |
| **Macro avg** | 0.936 | 0.922 | 0.927 | 27 |
| **Weighted avg** | 0.930 | 0.926 | 0.925 | 27 |

### 3.5 Confusion matrix (rows = actual, cols = predicted)

| | Pred class_0 | Pred class_1 | Pred class_2 |
|---|---|---|---|
| **Actual class_0** | 9 | 0 | 0 |
| **Actual class_1** | 1 | 10 | 0 |
| **Actual class_2** | 0 | 1 | 6 |

Generated artifacts: `training_history.png` (loss/accuracy curves), `confusion_matrix.png`, `classification_report.txt`, `metrics.json`.

---

## 4. Observations from this run

*(Starter observations below — add your own findings as you re-run with different hyperparameters, since that's the actual point of this exercise.)*

- **Overfitting shows up fast and clearly.** Training loss drops to near 0 by epoch ~6–10 and training accuracy hits 1.000, but validation loss bottoms out around epoch 3 (~0.099) and then *increases* even as training loss keeps falling. This gap between the two curves is the textbook signature of overfitting — the model starts memorizing training examples rather than learning generalizable patterns. Early stopping caught this and rolled back to the better epoch.
- **All two misclassifications happen between adjacent classes (1 and 2), not class 0.** class_0 was recovered perfectly (recall 1.000). This suggests class_0 is the most chemically distinct cultivar in this feature space, while classes 1 and 2 have some overlapping feature distributions.
- **Precision vs. recall trade-off across classes:** class_2 has perfect precision (1.000) but lower recall (0.857) — when the model predicts class_2 it's always right, but it misses some true class_2 samples (calling them class_1 instead). class_0 shows the opposite pattern (perfect recall, slightly lower precision).
- **Only 1,027 trainable parameters** were enough to reach ~93% test accuracy on this dataset — a reminder that model capacity should match dataset size/complexity; a much bigger network here would likely overfit even faster on only 178 total samples.
- **Standardization mattered.** Wine's 13 features are on very different scales (e.g. proline is in the hundreds, while some ratios are under 5). Without `StandardScaler`, features with larger raw magnitudes would dominate the first layer's weighted sums regardless of true importance.

### Suggested next experiments (for your own write-up)
- Remove `Dropout` and re-run — does validation loss overfit even faster?
- Try `batch_size=32` instead of 8 — does the loss curve get smoother or noisier?
- Try a lower learning rate (e.g. 0.001) — does it take longer to converge but overfit less?
- Remove `StandardScaler` entirely and compare accuracy — quantify how much scaling helped.
- Swap in a different dataset (e.g. `load_digits` or `load_breast_cancer`, adjusting `num_classes`) and compare how much harder/easier convergence is.

---

## 5. How to run

```bash
pip install tensorflow-cpu scikit-learn pandas matplotlib
python mlp_multiclass.py
```

Note: results (accuracy, confusion matrix) may shift slightly across machines/TensorFlow versions even with a fixed seed, due to floating-point non-determinism in some GPU/CPU ops — this is expected and worth noting in your own report rather than something to "fix."
