import streamlit as st
import numpy as np
import pandas as pd

# Set a random seed for consistent results across runs
np.random.seed(42)

# --- 1. Data and Network Setup ---

# Training data (raw)
X_raw = np.array([
    [30, 0.5, 50000],
    [35, 0.6, 55000],
    [40, 0.4, 60000],
    [45, 0.3, 65000]
])

# Normalize the input data to a scale of 0-1 for better training stability.
X = X_raw / np.max(X_raw, axis=0)

# Target values
y = np.array([
    [1],
    [1],
    [0],
    [0]
])

# Number of input features and output neurons
n_input = X.shape[1]
n_output = y.shape[1]

# --- 2. Core Neural Network Functions ---

def sigmoid(x):
    """The sigmoid activation function."""
    return 1 / (1 + np.exp(-x))

def forward(X, w):
    """The forward propagation function."""
    output = sigmoid(np.dot(X, w))
    return output

# --- 3. Training Process ---

def train(X, y, n_input, n_output, learning_rate, epochs):
    """
    The main training function.
    It repeatedly adjusts the weights to minimize the prediction error.
    """
    # Reinitialize weights at the start of training.
    # Initial weights must be passed to the function if we want to track them,
    # but for simplicity, we reinitialize here as in the original code.
    w = np.random.randn(n_input, n_output)

    # List to store error for plotting (optional, but good for interactive apps)
    # error_history = []

    for epoch in range(epochs):
        # Forward pass: make a prediction.
        output = forward(X, w)

        # Backpropagation: Calculate the error and the gradient.
        error = y - output
        # Gradient for Sigmoid: error * output * (1 - output)
        gradient = np.dot(X.T, error * output * (1 - output))

        # Update the weights.
        w += learning_rate * gradient

        # Calculate and store error (Mean Squared Error)
        # mse = np.mean(np.power(error, 2))
        # error_history.append(mse)

    # Final error calculation
    final_error = np.mean(np.power(y - forward(X, w), 2))
    return w, output, final_error

# ----------------- STREAMLIT UI SETUP -----------------

st.title("Simple Single-Layer Neural Network Demo")

st.markdown("""
This application demonstrates a basic single-layer neural network for binary classification.
Use the sidebar to adjust the training parameters and see the results update automatically.
""")
st.divider()

# --- 5. Streamlit Sidebar for Parameters ---
st.sidebar.header("Training Parameters")

# Sliders for user-defined parameters
LEARNING_RATE = st.sidebar.slider(
    "Learning Rate",
    min_value=0.0001,
    max_value=0.1,
    value=0.001,
    step=0.0001,
    format="%f"
)

EPOCHS = st.sidebar.slider(
    "Epochs (Training Iterations)",
    min_value=1000,
    max_value=50000,
    value=10000,
    step=1000
)

st.sidebar.markdown(f"**Data Summary:**")
st.sidebar.markdown(f"- Input Features: {n_input}")
st.sidebar.markdown(f"- Training Samples: {X.shape[0]}")


# --- 6. Run Training and Display Results ---

# Run the training function with user-defined parameters
final_w, predicted_result, final_error = train(
    X, y, n_input, n_output, LEARNING_RATE, EPOCHS
)

## Input Data Section
st.header("1. Input and Normalized Data")

# Display raw and normalized data side-by-side
col1, col2 = st.columns(2)

with col1:
    st.subheader("Raw Input Data (X_raw)")
    df_raw = pd.DataFrame(X_raw, columns=['Feature 1 (Age)', 'Feature 2 (Value)', 'Feature 3 (Salary)'])
    st.dataframe(df_raw, use_container_width=True)

with col2:
    st.subheader("Normalized Input Data (X)")
    df_norm = pd.DataFrame(X, columns=['Feature 1 (Norm)', 'Feature 2 (Norm)', 'Feature 3 (Norm)'])
    st.dataframe(df_norm, use_container_width=True)

st.subheader("Target Data (y)")
df_target = pd.DataFrame(y, columns=['Target Class (0 or 1)'])
st.dataframe(df_target, use_container_width=True)

st.divider()

## Results Section
st.header("2. Training Results")

st.metric(label="Final Mean Squared Error (MSE)", value=f"{final_error:.6f}")

st.subheader("Final Weights (w)")
st.write("These are the learned parameters of the network:")
df_weights = pd.DataFrame(final_w, index=['Feature 1 Weight', 'Feature 2 Weight', 'Feature 3 Weight'], columns=['Weight'])
st.dataframe(df_weights, use_container_width=True)

st.subheader("Predicted Output vs. Target")

# Combine target and prediction into a single DataFrame for easy viewing
results_df = pd.DataFrame({
    'Target (y)': y.flatten(),
    'Prediction (Sigmoid)': predicted_result.flatten()
})
# Add a 'Classification' column (e.g., probability > 0.5 is Class 1)
results_df['Binary Prediction'] = np.where(predicted_result > 0.5, 1, 0)
results_df['Correct?'] = np.where(results_df['Binary Prediction'] == results_df['Target (y)'], "✅ Correct", "❌ Incorrect")

st.dataframe(results_df, use_container_width=True)

st.caption("The **Prediction (Sigmoid)** column represents the network's confidence (probability between 0 and 1) for the positive class (1).")
