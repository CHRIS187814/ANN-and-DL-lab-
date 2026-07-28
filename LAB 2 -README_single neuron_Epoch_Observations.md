# Single Neuron (OR Gate) — Per-Epoch Output & Observations

## What Changed in the Code

Originally `train()` only printed the average loss, and only once every
100 epochs. It's now updated to print, **every single epoch**:

1. The average loss across all 4 training examples
2. The current prediction for each input, vs. its target

```python
if verbose:
    print(f"Epoch {epoch + 1:4d} | Avg Loss: {avg_loss:.6f}")
    for inputs, target in training_data:
        prediction = self.forward(inputs)
        print(f"    Input: {inputs} -> Predicted: {prediction:.4f} "
              f"| Target: {target}")
```

## Loss at Key Epochs (from an actual run)

| Epoch | Avg Loss |
|---|---|
| 1 | 0.191187 |
| 2 | 0.150410 |
| 5 | 0.110379 |
| 10 | 0.089742 |
| 20 | 0.065767 |
| 50 | 0.033254 |
| 100 | 0.016697 |
| 200 | 0.007825 |
| 300 | 0.004983 |
| 500 | 0.002830 |
| 700 | 0.001957 |
| 1000 | 0.001329 |

## Predictions at Start vs. End

| Input | Epoch 1 Prediction | Epoch 1000 Prediction | Target |
|---|---|---|---|
| [0, 0] | 0.4268 | 0.0546 | 0 |
| [0, 1] | 0.4892 | 0.9660 | 1 |
| [1, 0] | 0.6945 | 0.9659 | 1 |
| [1, 1] | 0.7451 | 0.9999 | 1 |

*(Exact numbers will differ slightly on your own run — weights start
random via `random.uniform(-1, 1)`, so the starting point isn't fixed
like the perceptron's zero-initialization.)*

---

## Observations

**1. Loss decreases every single epoch — smoothly, unlike the perceptron.**
Because this uses gradient descent on a continuous, differentiable loss
(MSE), the loss curve is monotonic — it never bounces up like the
perceptron's total error did. Every update strictly moves downhill.

**2. Progress is fast at first, then slows down a lot.**
Loss drops from `0.19 → 0.09` in the first 10 epochs, but takes until
epoch ~700 to get from `0.002` to `0.0019`. This is classic gradient
descent behavior: big gradients early (far from the solution) shrink as
the neuron approaches a good fit, so each step's improvement gets smaller.

**3. It never reaches exactly 0 loss.**
Unlike the perceptron (which hit *exactly* 0 total error and then froze),
sigmoid outputs can only approach 0 or 1 asymptotically — they mathematically
never touch them. So loss keeps inching down forever; it's never
"finished," just close enough. That's why this needs a fixed epoch count
instead of a "stop when converged" condition like the perceptron used.

**4. The hardest input to learn is [0, 0].**
Its target is 0, but sigmoid(z) starts around 0.4 (near the middle of the
curve, where the network is least confident). It needs the largest push
in the negative direction — and its final prediction (0.0546) still isn't
as close to 0 as [1,1]'s prediction (0.9999) is to 1. This is common:
whichever example is being over-predicted early takes longer to correct.

**5. Weights end up far from their random start.**
Both weights converge to roughly the same large positive value
(~6.2 each), and bias converges to a negative value (~ -2.85). This makes
intuitive sense for OR: you want *either* input alone to push `z` well
past 0 (large positive weights), while the negative bias cancels out the
case where both inputs are 0.

---
