import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression, make_blobs

# =============================================================================
# 1. Linear Regression with Gradient Descent
# =============================================================================
# GOAL: Find the best line (y = mx + c) to fit a set of data points.
# THE LOOP: We'll repeatedly adjust our slope (m) and intercept (c) to
# minimize the error between our line's predictions and the actual data.
# This iterative adjustment process is called Gradient Descent.
# -----------------------------------------------------------------------------

def linear_regression_loop(X, y, learning_rate=0.01, n_iterations=1000):
    """
    Implements Linear Regression using a gradient descent loop.

    Args:
        X (np.array): Feature data (independent variable).
        y (np.array): Target data (dependent variable).
        learning_rate (float): How big of a step to take on each iteration.
        n_iterations (int): How many times to loop and adjust the parameters.

    Returns:
        tuple: The final slope (m) and intercept (c).
    """
    print("--- Starting Linear Regression Loop ---")
    # Step 1: Initialize parameters randomly
    n_samples, n_features = X.shape
    # We use n_features for weights (m) and a single bias (c)
    m_weights = np.zeros(n_features)
    c_bias = 0

    # This is the core "learning loop"!
    for i in range(n_iterations):
        # Step 2: Make a prediction with the current m and c
        y_predicted = np.dot(X, m_weights) + c_bias

        # Step 3: Calculate the error (how wrong our prediction is)
        # We use Mean Squared Error (MSE)
        cost = (1/n_samples) * np.sum((y_predicted - y)**2)

        # Step 4: Calculate the derivatives (the direction to adjust our parameters)
        # This is the "gradient" part of gradient descent
        dm = (2/n_samples) * np.dot(X.T, (y_predicted - y))
        dc = (2/n_samples) * np.sum(y_predicted - y)

        # Step 5: Update the parameters to reduce the error
        # We move in the opposite direction of the gradient
        m_weights -= learning_rate * dm
        c_bias -= learning_rate * dc

        if i % 100 == 0:
            print(f"Iteration {i}: Cost = {cost:.4f}")

    print("--- Linear Regression Loop Finished ---")
    return m_weights, c_bias

# --- Example Usage for Linear Regression ---
# Generate some sample data
X_reg, y_reg = make_regression(n_samples=100, n_features=1, noise=20, random_state=42)

# Run the algorithm
m_final, c_final = linear_regression_loop(X_reg, y_reg)
print(f"\nFinal Parameters: Slope (m) = {m_final[0]:.4f}, Intercept (c) = {c_final:.4f}\n")

# Plot the results
plt.figure(figsize=(8, 6))
plt.scatter(X_reg, y_reg, color='blue', label='Actual Data')
plt.plot(X_reg, np.dot(X_reg, m_final) + c_final, color='red', linewidth=3, label='Fitted Line')
plt.title('Linear Regression from Scratch')
plt.xlabel('Feature (X)')
plt.ylabel('Target (y)')
plt.legend()
plt.show()