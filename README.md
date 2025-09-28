# Interactive Single-Layer Neural Network (Streamlit App)

This repository contains a simple, single-neuron neural network implemented from scratch in Python using **NumPy**. It has been upgraded with a **Streamlit user interface** to create an interactive learning and demonstration tool.

The purpose of this project is to demonstrate the fundamental concepts of a neural network, including **forward propagation**, **backpropagation**, and **gradient descent**, in a clean, understandable way.

***

## Project Overview and Features

The application provides an interactive environment where you can explore how a basic neural network learns from data:

* **Interactive Parameters:** You can easily change the **Learning Rate** and the number of **Epochs** (training iterations) using sliders in the interface.
* **Real-time Results:** As you adjust the parameters, the network **trains instantly**, and you can see the updated **Final Weights**, **Predictions**, and the final **Mean Squared Error (MSE)**.
* **Data Visualization:** The app displays the original **Raw Input Data**, the **Normalized Data** used for training, and the **Target Values** (the correct answers).

***

## Core Neural Network Concepts Demonstrated

This simple network is designed for a **binary classification** task, learning to predict one of two possible outcomes (0 or 1) based on three input features.

### 1. Data Preparation
* **Normalization:** The raw input data (which includes large numbers like salaries) is scaled down to values between 0 and 1. This prevents features with large magnitudes from unfairly dominating the learning process, ensuring more stable and efficient training.

### 2. Network Components
* **Weights:** These are the parameters the network *learns*. They represent the strength of the connection between the input features and the output.
* **Sigmoid Function:** This is the network's **activation function**. It "squashes" the output of the weighted sum into a value between 0 and 1, allowing the result to be interpreted as a **probability** for the positive class (1).

### 3. Training Process
* **Forward Propagation:** The input data is passed through the network, multiplying the inputs by the current weights and applying the Sigmoid function to produce a prediction.
* **Backpropagation & Gradient Descent:** This is the learning engine.
    * The **error** (the difference between the prediction and the target) is calculated.
    * **Backpropagation** determines how much each weight contributed to that error.
    * **Gradient Descent** uses this information to slightly adjust the weights in the direction that will minimize the error in the next iteration.
* **Epochs:** The network repeats the forward and backpropagation steps for a specified number of epochs (e.g., 10,000) to refine its weights and improve accuracy.

### Screenshots
<img width="2559" height="1239" alt="image" src="https://github.com/user-attachments/assets/2f02c6ed-69e2-4490-8e07-c2547bd9c3cf" />
<img width="2559" height="1191" alt="image" src="https://github.com/user-attachments/assets/f7e8c136-9974-49a1-aff9-c5b85168f964" />
<img width="2555" height="1201" alt="image" src="https://github.com/user-attachments/assets/82379448-5085-4c80-bc00-0b391562787a" />
