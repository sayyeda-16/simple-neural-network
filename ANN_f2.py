import numpy as np

# --- 1. Data and Network Setup ---

# Number of input features for each data point
n_input = 3

# Number of output neurons (1 for binary classification)
n_output = 1

# Set a random seed for consistent results across runs
np.random.seed(42)

# Training data. Each row is a data point with 3 features (e.g., age, a value, salary)
X = np.array([
    [30, 0.5, 50000],
    [35, 0.6, 55000],
    [40, 0.4, 60000],
    [45, 0.3, 65000]
])

# Normalize the input data to a scale of 0-1 for better training stability.
# This prevents features with large values (like salary) from dominating the learning.
X = X / np.max(X, axis=0)

# Target values (the correct answers) for each corresponding row in X
# 1 represents one class, 0 represents another.
y = np.array([
    [1],
    [1],
    [0],
    [0]
])

# Initialize the weights of the neural network with random values.
# These weights are what the network will learn during training.
w = np.random.randn(n_input, n_output)

# --- 2. Core Neural Network Functions ---

# The sigmoid activation function. It "squashes" any input value to a range between 0 and 1,
# which is useful for interpreting the output as a probability.
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# The forward propagation function.
# It takes the input data and weights, and computes the network's output prediction.
def forward(X, w):
    # The core operation: matrix multiplication of inputs and weights, followed by the sigmoid function.
    output = sigmoid(np.dot(X, w))
    return output

# --- 3. Training Process ---

# The main training function. It repeatedly adjusts the weights to minimize the prediction error.
def train(X, y, learning_rate=0.001, epochs=10000):
    # Reinitialize weights at the start of training.
    w = np.random.randn(n_input, n_output)

    # Loop through the training process for a specified number of epochs.
    for epoch in range(epochs):
        # Forward pass: make a prediction.
        output = forward(X, w)

        # Backpropagation: Calculate the error and the gradient (the direction to change the weights).
        error = y - output
        # The gradient is calculated using the derivative of the sigmoid function.
        gradient = np.dot(X.T, error * output * (1 - output))

        # Update the weights. The learning rate controls how big of a step we take.
        w += learning_rate * gradient

    return w, output

# --- 4. Running the Code ---

# Start the training process and get the final weights and predicted results.
w, predicted_result = train(X, y)

# Calculate the mean squared error to evaluate the network's performance.
error = np.mean(np.power(y - predicted_result, 2))

# Print the results
print("Weights:\n", w)
print("\nPredicted result: ", predicted_result)
print("Error: ", error)