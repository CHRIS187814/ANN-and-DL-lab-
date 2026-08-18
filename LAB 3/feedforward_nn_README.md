# feedforward_nn.py

A simple feedforward neural network (multi-layer perceptron) implemented in **PyTorch**, trained to classify handwritten digits from the **MNIST** dataset.

## Overview

- **Framework:** PyTorch
- **Task:** Digit classification (0–9)
- **Dataset:** MNIST (60,000 train / 10,000 test images, 28x28 grayscale)
- **Architecture:** Fully connected network
  - Input layer: 784 units (flattened 28x28 image)
  - Hidden layer 1: 256 units, ReLU, Dropout(0.2)
  - Hidden layer 2: 128 units, ReLU, Dropout(0.2)
  - Output layer: 10 units (one per digit class)
- **Loss function:** Cross-Entropy Loss
- **Optimizer:** Adam (lr = 0.001)
- **Epochs:** 10
- **Batch size:** 64

## Files

| File | Description |
|---|---|
| `feedforward_nn.py` | Main script: loads data, defines model, trains, evaluates, plots results |
| `training_curves.png` | Generated after running the script — loss & accuracy curves |
| `data/` | Auto-downloaded MNIST dataset (created on first run) |

## Requirements

```bash
pip install torch torchvision matplotlib
```

## How to Run

```bash
python feedforward_nn.py
```

On first run, MNIST will be automatically downloaded into a local `./data` folder.

## Expected Output

The script prints per-epoch training/test loss and accuracy, e.g.:

```
Epoch [1/10] Train Loss: 0.3421, Train Acc: 0.8985 | Test Loss: 0.1583, Test Acc: 0.9521
...
Epoch [10/10] Train Loss: 0.0512, Train Acc: 0.9843 | Test Loss: 0.0721, Test Acc: 0.9789

Final Test Accuracy: 97.89%
```

It also saves a plot, `training_curves.png`, showing loss and accuracy curves for both training and test sets across epochs.

## Notes / Possible Extensions

- Swap `HIDDEN_SIZES` in the script to experiment with network depth/width.
- Increase `EPOCHS` or tune `LEARNING_RATE` to see effects on convergence.
- Replace the MNIST dataset with Fashion-MNIST or CIFAR-10 for a harder classification task.
- A TensorFlow/Keras version can be provided on request if required for comparison.
