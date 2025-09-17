# Simple Neural Network

This repository contains a simple, single-neuron neural network implemented from scratch in Python using NumPy. The purpose of this project is to demonstrate the fundamental concepts of a neural network, including **forward propagation**, **backpropagation**, and **gradient descent**, in a clean and understandable way.

---

### Key Concepts Demonstrated

* **Normalization:** The input data is normalized to ensure training stability and prevent features with larger values (like salary) from dominating the learning process.
* **Activation Function:** The **sigmoid function** is used to introduce non-linearity and to squash the output to a value between 0 and 1, making it suitable for binary classification.
* **Forward Propagation:** The process of feeding input data through the network to generate a prediction.
* **Backpropagation:** The core training algorithm that calculates the error and determines how much to adjust the network's weights.
* **Gradient Descent:** The method used to update the weights in the direction that minimizes the prediction error.

---

### How It Works

The network is designed for a **binary classification** task. It takes three input features and learns to predict one of two possible outputs (0 or 1). The `train` function iteratively adjusts the network's weights over 10,000 epochs to make its predictions as close to the target values as possible.

---


