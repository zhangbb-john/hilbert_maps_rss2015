import numpy as np
import matplotlib.pyplot as plt

# Sigmoid function for logistic regression
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Logistic regression cost function (negative log-likelihood)
def compute_cost(X, y, w, epsilon=1e-15):
    m = len(y)
    h = sigmoid(np.dot(X, w))
    # Clip h to avoid log(0) and numerical issues
    h = np.clip(h, epsilon, 1 - epsilon)
    cost = - (1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    return cost

# Gradient of the logistic regression cost function
def compute_gradient(X, y, w):
    m = 1
    h = sigmoid(np.dot(X, w))
    gradient = (1/m) * np.dot(X.T, (h - y))  # Derivative of log-likelihood
    return gradient

# Stochastic Gradient Descent (SGD) for logistic regression
def sgd_logistic_regression(X, y, alpha=0.01, t0=1, max_iter=10):
    m, n = X.shape
    w = np.zeros(n)  # Initialize parameters (weights) to zero
    cost_history = []

    # SGD updates
    for t in range(1, max_iter + 1):
        # Pick a random sample
        i = np.random.randint(0, m)
        xi = X[i, :].reshape(1, -1)  # Ensure xi is a 2D array (1 sample)
        yi = y[i]
        
        # Compute gradient and update the parameters
        gradient = compute_gradient(xi, yi, w)  # Use the full input sample xi
        # Update rule: w_t = w_(t-1) - eta_t * gradient
        eta_t = 1 / (alpha * (t0 + t))  # Learning rate decreases over time
        w = w - eta_t * gradient
        
        # Compute the cost at every 10 iterations for plotting
        if t % 10 == 0:  # Save cost every 10 iterations for more frequent updates
            cost = compute_cost(X, y, w)
            cost_history.append(cost)
    
    return w, cost_history

# Example usage with synthetic data

# Generate some synthetic data for binary classification
np.random.seed(42)
m = 200  # Number of samples
n = 2    # Number of features
X = np.random.randn(m, n)  # Random feature matrix
y = (np.dot(X, np.array([2, -3])) + 0.5 > 0).astype(int)  # Labels based on a linear function

# Add intercept term (bias) to the input data X
X = np.hstack([np.ones((X.shape[0], 1)), X])  # Adding a column of ones for the intercept term

# Train the model using SGD
w_opt, cost_history = sgd_logistic_regression(X, y, alpha=0.01, t0=1, max_iter=2)

# Display results
print("Optimized weights:", w_opt)

# Plot the cost history to visualize convergence
plt.figure(1)  # Create a new figure for the cost plot
plt.plot(cost_history)
plt.xlabel('Iterations (every 10 steps)')
plt.ylabel('Cost (Negative Log-Likelihood)')
plt.title('Convergence of SGD for Logistic Regression')
plt.grid(True)

# Classify the data using the learned weights
def predict(X, w):
    h = sigmoid(np.dot(X, w))
    return (h >= 0.5).astype(int)  # Return 1 if probability is >= 0.5, else 0

# Make predictions on the training set
y_pred = predict(X, w_opt)

# Visualize the classification performance

# Plot original data points with true labels
plt.figure(2)  # Create a new figure for the classification plot
plt.scatter(X[:, 1], X[:, 2], c=y, cmap='coolwarm', label='True Labels', alpha=0.3, s = 8)
# Add labels, title, and legend
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Logistic Regression Classification: True vs Predicted Labels')
plt.legend()
plt.grid(True)

plt.figure(3)  # Create a new figure for the classification plot
# Plot data points with predicted labels
plt.scatter(X[:, 1], X[:, 2], c=y_pred, cmap='coolwarm', marker='x', label='Predicted Labels', s=10, alpha=0.7)

# Add labels, title, and legend
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Logistic Regression Classification: True vs Predicted Labels')
plt.legend()
plt.grid(True)
plt.show()
