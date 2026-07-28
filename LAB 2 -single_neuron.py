"""
single_neuron.py
-----------------
A single artificial neuron implemented manually (no ML libraries like
TensorFlow/PyTorch/sklearn). Only Python's built-in `math` and `random`
modules are used, so every calculation is visible and explicit.

The neuron learns the logical OR function as a demonstration.
See README.md for a term-by-term explanation of every keyword used here.
"""

import random
import math


class Neuron:
    def __init__(self, num_inputs, learning_rate=0.1):
        # weights: one value per input feature, randomly initialized
        self.weights = [random.uniform(-1, 1) for _ in range(num_inputs)]
        # bias: a single extra trainable value, independent of inputs
        self.bias = random.uniform(-1, 1)
        # learning_rate: controls how big each training update step is
        self.learning_rate = learning_rate

    def weighted_sum(self, inputs):
        # z = (w1*x1 + w2*x2 + ... + wn*xn) + b
        z = sum(w * x for w, x in zip(self.weights, inputs)) + self.bias
        return z

    def activate(self, z):
        # sigmoid activation function: squashes any real number into (0, 1)
        return 1 / (1 + math.exp(-z))

    def activate_derivative(self, activated_output):
        # derivative of sigmoid, expressed in terms of its own output
        # used during backpropagation to compute gradients
        return activated_output * (1 - activated_output)

    def forward(self, inputs):
        # forward pass / forward propagation:
        # inputs -> weighted sum -> activation -> prediction
        z = self.weighted_sum(inputs)
        output = self.activate(z)
        return output

    def train_step(self, inputs, target):
        # ---- forward pass ----
        prediction = self.forward(inputs)

        # ---- loss calculation ----
        # mean squared error (MSE) for a single sample
        error = target - prediction
        loss = error ** 2

        # ---- backward pass (backpropagation) ----
        # gradient of loss w.r.t. the neuron's output
        d_loss_d_pred = -2 * error
        # gradient of activation w.r.t. the weighted sum z
        d_pred_d_z = self.activate_derivative(prediction)
        # combined gradient of loss w.r.t. z (chain rule)
        d_loss_d_z = d_loss_d_pred * d_pred_d_z

        # ---- gradient descent update ----
        # update each weight: w = w - learning_rate * gradient
        for i in range(len(self.weights)):
            gradient_w = d_loss_d_z * inputs[i]
            self.weights[i] -= self.learning_rate * gradient_w

        # update bias the same way (its "input" is always 1)
        gradient_b = d_loss_d_z * 1
        self.bias -= self.learning_rate * gradient_b

        return loss

    def train(self, training_data, epochs=1000, verbose=True):
        # training_data: list of (inputs, target) pairs
        # epoch: one full pass through the entire training dataset
        for epoch in range(epochs):
            total_loss = 0
            for inputs, target in training_data:
                total_loss += self.train_step(inputs, target)
            avg_loss = total_loss / len(training_data)

            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch:4d} | Avg Loss: {avg_loss:.6f}")


if __name__ == "__main__":
    # Training data for the OR logic gate:
    # (inputs, target_output)
    training_data = [
        ([0, 0], 0),
        ([0, 1], 1),
        ([1, 0], 1),
        ([1, 1], 1),
    ]

    neuron = Neuron(num_inputs=2, learning_rate=0.5)

    print("Training single neuron on OR gate...\n")
    neuron.train(training_data, epochs=1000)

    print("\nFinal weights:", neuron.weights)
    print("Final bias:", neuron.bias)

    print("\nPredictions after training:")
    for inputs, target in training_data:
        prediction = neuron.forward(inputs)
        print(f"Input: {inputs} -> Predicted: {prediction:.4f} | Target: {target}")
