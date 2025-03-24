# from sklearn.kernel_approximation import RBFSampler
# from sklearn.linear_model import SGDClassifier
# import numpy as np

# # Generate some toy data
# X = np.random.rand(10, 2)  # 10 samples, 2 features
# y = np.random.choice([0, 1], size=10)  # Binary labels

# # Step 1: Approximate RBF kernel
# rbf_feature = RBFSampler(gamma=1.0, random_state=42)  # gamma controls the kernel width
# X_features = rbf_feature.fit_transform(X)  # Transform input data

# # Step 2: Use transformed features in a linear classifier
# clf = SGDClassifier(loss='hinge', random_state=42)  # Linear SVM
# clf.fit(X_features, y)  # Train classifier

# # Predict on new data
# X_new = np.random.rand(5, 2)  # New samples
# X_new_features = rbf_feature.transform(X_new)  # Apply the same RBF transformation
# predictions = clf.predict(X_new_features)  # Predict labels

# print("Predictions:", predictions)



from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
# Visualization 3: Decision Boundary in Original Space
def plot_decision_boundary(X, y, model, rbf_feature, title):
    h = 0.02  # step size for the mesh
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(rbf_feature.transform(grid_points))
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8, cmap="coolwarm")
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k", cmap="coolwarm")
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
# Generate some toy data
np.random.seed(42)
X_train = np.random.rand(50, 2)  # Training data: 50 samples, 2 features
y_train = np.random.choice([0, 1], size=50)  # Binary labels for training
X_test = np.random.rand(20, 2)  # Test data: 20 samples, 2 features
y_test = np.random.choice([0, 1], size=20)  # Binary labels for testing

# Initialize RBFSampler
rbf_feature = RBFSampler(gamma=1.0, random_state=42)

# Step 1: Fit on training data
rbf_feature.fit(X_train)  # Learn random Fourier components

# Step 2: Transform training and test data
X_train_features = rbf_feature.transform(X_train)  # Transform training data
X_test_features = rbf_feature.transform(X_test)  # Transform test data using the same mapping

# Step 3: Train a linear classifier on the transformed data
clf = SGDClassifier(loss='log', random_state=42)  # Logistic regression
clf.fit(X_train_features, y_train)  # Train on transformed training data

# Step 4: Make predictions on the transformed test data
y_pred = clf.predict(X_test_features)  # Predict on test data
print("Predicted values: ", y_pred)
# Step 5: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy
print("Accuracy on test data:", accuracy)

# Add Visualizations Here
# Visualization 1: Scatter plot of original training data
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap="coolwarm", edgecolor="k")
plt.title("Original Training Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
print("Plotted")
# Visualization 2: Scatter plot of transformed training data (reduce dimensionality to 2D) 
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
print("X_train_features: ", X_train_features.size)
#Since RBF transformation results in high-dimensional features, PCA reduces them to 2D for visualization.

X_train_pca = pca.fit_transform(X_train_features)

plt.subplot(1, 3, 2)
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap="coolwarm", edgecolor="k")
plt.title("Transformed Training Data (PCA)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")



plt.subplot(1, 3, 3)
plot_decision_boundary(X_train, y_train, clf, rbf_feature, "Decision Boundary (RBF Approx.)")

plt.tight_layout()
plt.show()
