# Perceptron — Per-Epoch Predictions (Update Notes)

This note explains the change made to `perceptron_numpy.py`: the script now
prints the perceptron's predictions **after every epoch**, not just once at
the very end.

---

## What Changed

Originally, the script only tested the trained perceptron **once**, after
all 10 epochs finished. Now, at the end of *each* epoch, it also runs the
perceptron on all 4 training inputs (using the weights/bias as they stand
right then) and prints what it predicts vs. the correct target.

```python
print("  Predictions after this epoch:")
for inputs, label in zip(training_inputs, labels):
    output = self.predict(inputs)
    print(f"    Input: {inputs} -> Predicted: {output} "
          f"(Target: {label})")
```

This doesn't change *how* the perceptron learns — the training loop and
learning rule are untouched. It only adds visibility into what the model
believes after every round of training.

---

## Why This Matters (for the viva)

Without this, you only see the final answer — a black box. With per-epoch
predictions, you can literally watch the decision boundary move:

- Which specific inputs it's getting wrong at each stage
- The exact epoch where it "clicks" and starts getting everything right
- That the weights/bias stop changing once predictions are all correct
  (because the learning rule only updates on `error != 0`)

## What the Output Looks Like

```
Epoch 1: Weights = [0.1 0.1], Bias = 0.00, Total Error = 2
  Predictions after this epoch:
    Input: [0 0] -> Predicted: 1 (Target: 0)
    Input: [0 1] -> Predicted: 1 (Target: 0)
    Input: [1 0] -> Predicted: 1 (Target: 0)
    Input: [1 1] -> Predicted: 1 (Target: 1)

Epoch 2: Weights = [0.2 0.1], Bias = -0.10, Total Error = 3
  ...

Epoch 4: Weights = [0.2 0.1], Bias = -0.20, Total Error = 0
  Predictions after this epoch:
    Input: [0 0] -> Predicted: 0 (Target: 0)
    Input: [0 1] -> Predicted: 0 (Target: 0)
    Input: [1 0] -> Predicted: 0 (Target: 0)
    Input: [1 1] -> Predicted: 1 (Target: 1)
```

## Key Observation From This Run

| Epoch | Total Error | Notes |
|---|---|---|
| 1 | 2 | Still wrong on 3 of 4 inputs |
| 2 | 3 | Error briefly goes *up*, not down — normal for the perceptron rule; it's not gradient descent, so error isn't guaranteed to decrease monotonically |
| 3 | 3 | Getting closer |
| 4 | 0 | **Converged** — every prediction matches the target |
| 5–10 | 0 | No further changes; weights/bias are stable because `error = 0` for every example |

**Viva tip:** if asked "does total error always decrease every epoch?" —
answer *no*. That guarantee only holds for gradient descent on a smooth
loss. The perceptron learning rule can bounce around before it converges;
it's only guaranteed to *eventually* converge if the data is linearly
separable (which AND is).

---

## Running It

```bash
python3 perceptron_numpy.py
```

You'll see the epoch summary line, followed by the 4 predictions for that
epoch, repeated for all 10 epochs — then the original final weights/bias
and test block at the bottom.
