# Lab 1: Simple Perceptron Simulation (NumPy)

This program implements a **perceptron** — the simplest form of an
artificial neuron — using **NumPy** for vector/array math. The perceptron
is trained to learn the **AND logic gate**.

---

## Files

| File | Purpose |
|---|---|
| `perceptron_numpy.py` | The `Perceptron` class, training loop, and a demo run |
| `README_Lab1_Perceptron.md` | This file — explains every term/keyword used in the code |

---

## How a Perceptron Works (Concept)

A perceptron takes inputs, multiplies each by a weight, adds a bias, and
passes the result through a **step function** to produce a strictly binary
output (`0` or `1`):

```
inputs -> weighted sum -> step function -> output (0 or 1)
```

Whenever the perceptron's prediction is wrong, it nudges its weights and
bias slightly in the direction that would have made the prediction
correct. Repeating this over the training data lets it converge on a rule
that correctly separates the classes.

---

## Glossary of Terms Used in the Code

### `Perceptron` (class)
The object representing the perceptron. It stores the learnable
parameters (weights, bias) and contains the logic to predict and learn.

### `numpy` / `np`
A Python library for fast numerical computing with arrays. Here it's used
to store `weights` and `inputs` as arrays and to compute their dot product
efficiently, instead of writing manual loops.

### `weights`
An array of numbers, one per input feature, representing how strongly
each input influences the output. Initialized to `0` here, and updated
during training.

### `bias`
An extra learnable value added to the weighted sum, independent of the
inputs. It shifts the decision boundary, letting the perceptron fit data
that doesn't pass through the origin.

### `learning_rate`
A small constant (`0.1` here) controlling the size of each weight/bias
update during training. Higher values learn faster but can overshoot;
lower values are more stable but slower.

### `np.dot(self.weights, inputs)`
Computes the **dot product** of the weights and inputs — i.e.
`w1*x1 + w2*x2 + ... + wn*xn` — in one efficient operation. This is the
core linear calculation of the perceptron.

### `z` (Weighted Sum)
The result of `np.dot(weights, inputs) + bias`. It's a single number
summarizing all inputs before the activation function is applied.

### `step_function()` / Activation Function
The function applied to `z` to produce the final output:
```
step(z) = 1 if z >= 0
        = 0 if z <  0
```
Unlike the sigmoid used in a standard neuron (which outputs a smooth
probability between 0 and 1), the step function gives a hard, binary
decision — the defining trait of a classic perceptron.

### `predict()`
Runs one **forward pass**: takes inputs, computes the weighted sum,
applies the step function, and returns the predicted output.

### `error`
The difference between the correct label and the predicted output:
```
error = label - prediction
```
It can be `-1`, `0`, or `1` since both `label` and `prediction` are binary.
An error of `0` means the prediction was correct, so no update occurs.

### Perceptron Learning Rule
The update rule used to adjust weights and bias whenever a prediction is
wrong:
```
weights = weights + learning_rate * error * inputs
bias    = bias    + learning_rate * error
```
This is simpler than gradient descent with backpropagation (used in
sigmoid-based neurons) because the step function has no useful derivative
— the perceptron instead updates directly based on the sign of the error.

### `epoch`
One complete pass through the entire training dataset. This program runs
for `10` epochs, though the perceptron typically converges (reaches `0`
total error) much earlier if the data is linearly separable.

### `total_error`
The sum of absolute errors across all training examples in one epoch.
Used here just to monitor progress — once it hits `0`, the perceptron has
correctly classified every training example.

### `training_inputs` and `labels`
The dataset used for training — the truth table for the AND gate:

| Input A | Input B | AND Output |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 0 |
| 1 | 0 | 0 |
| 1 | 1 | 1 |

---

## Running the Code

```bash
python3 perceptron_numpy.py
```

Expected behavior: `Total Error` decreases each epoch and reaches `0`
within a few epochs (AND is linearly separable), after which the
weights and bias stop changing. Final predictions should exactly match
the AND gate truth table.

---

## Perceptron vs. the Sigmoid Neuron (from the earlier lab)

| | Perceptron | Sigmoid Neuron |
|---|---|---|
| Activation | Step function (hard 0/1) | Sigmoid (smooth 0 to 1) |
| Output type | Binary decision | Probability-like value |
| Learning rule | Perceptron rule (based on error sign) | Gradient descent (based on loss derivative) |
| Can learn XOR? | No | No (a single neuron of either type can't) |

Both are limited to **linearly separable** problems. Learning something
like XOR requires stacking multiple neurons into layers — a full neural
network.
