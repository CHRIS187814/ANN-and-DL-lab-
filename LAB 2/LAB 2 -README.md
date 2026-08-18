# Single Neuron Model (Manual Implementation)

This project implements a single artificial neuron **from scratch in Python**,
using only the built-in `math` and `random` modules — no NumPy, TensorFlow,
PyTorch, or scikit-learn. The goal is to make every calculation transparent,
so you can see exactly how a neuron computes a prediction and learns from data.

The neuron is trained to learn the **OR logic gate**.

---

## Files

| File | Purpose |
|---|---|
| `single_neuron.py` | The neuron class, training loop, and a demo run |
| `README.md` | This file — explains every term/keyword used in the code |

---

## How a Single Neuron Works (Concept)

A neuron takes several inputs, multiplies each by a weight, adds a bias,
and passes the result through an activation function to produce an output:

```
inputs -> weighted sum -> activation function -> output (prediction)
```

During training, the neuron compares its prediction to the correct answer,
measures how wrong it was, and adjusts its weights and bias slightly to
reduce that error. Repeating this many times lets it "learn."

---

## Glossary of Terms Used in the Code

### `Neuron` (class)
The core object representing one artificial neuron. It holds the neuron's
learnable parameters (weights, bias) and the logic to make predictions and
learn from data.

### `weights`
A list of numbers, one per input feature. Each weight represents how much
influence that particular input has on the neuron's output. Larger weights
mean stronger influence. These values start random and are updated during
training.

### `bias`
An extra learnable value added to the weighted sum, independent of any
input. It lets the neuron shift its decision boundary — similar to the
y-intercept `b` in the line equation `y = mx + b`.

### `learning_rate`
A small number (e.g. `0.1` or `0.5`) that controls how large each update
step is during training. Too high, and training becomes unstable; too low,
and training is very slow.

### `weighted_sum()` / `z`
The core linear calculation of the neuron:
```
z = (w1*x1 + w2*x2 + ... + wn*xn) + bias
```
This is just a weighted combination of the inputs — no non-linearity yet.

### `activate()` / Activation Function
A function applied to `z` that introduces **non-linearity**, letting the
neuron model more complex relationships than a straight line. This project
uses the **sigmoid function**:
```
sigmoid(z) = 1 / (1 + e^(-z))
```
Sigmoid squashes any real number into a range between `0` and `1`, which is
useful for representing probabilities (e.g. "how likely is this a 1?").

### `activate_derivative()`
The mathematical derivative (rate of change) of the sigmoid function.
It's needed to calculate gradients during learning. Conveniently, the
derivative of sigmoid can be written using its own output:
```
sigmoid'(x) = x * (1 - x)
```

### `forward()` / Forward Pass (Forward Propagation)
The process of pushing inputs through the neuron to get a prediction:
`inputs -> weighted_sum -> activate -> output`. No learning happens here —
it's just a calculation.

### `loss` / Loss Function
A number that measures how wrong a prediction is compared to the target
(correct) value. This project uses **Mean Squared Error (MSE)**:
```
loss = (target - prediction)^2
```
A loss of `0` means a perfect prediction; higher values mean bigger errors.

### `train_step()` / Backward Pass (Backpropagation)
The process of calculating **how much each weight and the bias contributed
to the error**, using calculus (the chain rule). This tells the neuron
which direction to adjust each parameter to reduce the loss.

### `gradient`
A value representing the slope (direction and steepness) of the loss with
respect to a specific parameter (a weight or the bias). If the gradient is
positive, decreasing that parameter reduces the loss, and vice versa.

### Gradient Descent
The optimization technique used to update weights and bias:
```
parameter = parameter - (learning_rate * gradient)
```
This nudges each parameter a small step in the direction that reduces the
loss. Repeating this over many examples gradually improves the neuron's
predictions.

### `epoch`
One complete pass through the entire training dataset. Training usually
requires many epochs (this demo uses 1000) for the loss to shrink close
to zero.

### `training_data`
A list of `(inputs, target)` pairs — example inputs paired with the correct
answers the neuron should learn to predict. Here, it's the truth table for
the OR gate:

| Input A | Input B | OR Output |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 1 |

---

## Running the Code

```bash
python3 single_neuron.py
```

Expected behavior: the printed loss decreases every 100 epochs, and by the
end, the neuron's predictions closely match the target OR gate outputs
(values near `0` for target `0`, and near `1` for target `1`).

---

## Why a Single Neuron Has Limits

A single neuron with a sigmoid activation can only learn **linearly
separable** patterns (like OR, AND). It **cannot** learn XOR, because no
single straight line can separate XOR's outputs. Solving XOR requires
stacking multiple neurons into layers — which is the basis of neural
networks.
