"""
Lab 1: Simple Perceptron Simulation using NumPy
--------------------------------------------------
A perceptron is the simplest type of artificial neuron. It takes inputs,
multiplies them by weights, adds a bias, and passes the result through a
step activation function to produce a binary output (0 or 1).

This program trains a perceptron to learn the AND logic gate.
"""

import numpy as np


class Perceptron:
    def __init__(self, num_inputs, learning_rate=0.1):
        # Initialize weights and bias to zero (common for perceptrons)
        self.weights = np.zeros(num_inputs)
        self.bias = 0.0
        self.learning_rate = learning_rate

    def step_function(self, z):
        # Activation function: outputs 1 if z >= 0, else 0
        return 1 if z >= 0 else 0

    def predict(self, inputs):
        # Weighted sum: z = w.x + b
        z = np.dot(self.weights, inputs) + self.bias
        return self.step_function(z)

    def train(self, training_inputs, labels, epochs=10):
        for epoch in range(epochs):
            total_error = 0
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                error = label - prediction
                total_error += abs(error)

                # Perceptron learning rule:
                # w = w + learning_rate * error * inputs
                # b = b + learning_rate * error
                self.weights += self.learning_rate * error * inputs
                self.bias += self.learning_rate * error

            print(f"Epoch {epoch + 1}: Weights = {self.weights}, "
                  f"Bias = {self.bias:.2f}, Total Error = {total_error}")


if __name__ == "__main__":
    # Training data for the AND logic gate
    training_inputs = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    labels = np.array([0, 0, 0, 1])  # AND gate outputs

    perceptron = Perceptron(num_inputs=2, learning_rate=0.1)

    print("Training Perceptron on AND gate...\n")
    perceptron.train(training_inputs, labels, epochs=10)

    print("\nFinal weights:", perceptron.weights)
    print("Final bias:", perceptron.bias)

    print("\nTesting trained perceptron:")
    for inputs in training_inputs:
        output = perceptron.predict(inputs)
        print(f"Input: {inputs} -> Predicted Output: {output}")
