import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. XOR Dataset
# -----------------------------
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

# -----------------------------
# 2. Sigmoid Activation
# -----------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return z * (1 - z)

# -----------------------------
# 3. Initialize Parameters
# -----------------------------
np.random.seed(42)

input_size = 2
hidden_size = 4
output_size = 1

W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))

W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

learning_rate = 0.1
epochs = 5000

losses = []
accuracies = []

# -----------------------------
# 4. Training (Backpropagation)
# -----------------------------
for epoch in range(epochs):

    # Forward Propagation
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)

    z2 = np.dot(a1, W2) + b2
    y_pred = sigmoid(z2)

    # Loss (Binary Cross Entropy)
    loss = -np.mean(y*np.log(y_pred+1e-8) + (1-y)*np.log(1-y_pred+1e-8))
    losses.append(loss)

    # Accuracy
    predictions = (y_pred > 0.5).astype(int)
    accuracy = np.mean(predictions == y)
    accuracies.append(accuracy)

    # Backpropagation
    dz2 = y_pred - y
    dW2 = np.dot(a1.T, dz2)
    db2 = np.sum(dz2, axis=0, keepdims=True)

    dz1 = np.dot(dz2, W2.T) * sigmoid_derivative(a1)
    dW1 = np.dot(X.T, dz1)
    db1 = np.sum(dz1, axis=0, keepdims=True)

    # Update Weights
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

# -----------------------------
# 5. Final Predictions
# -----------------------------
print("Final Predictions:")
print(predictions)

# -----------------------------
# 6. Plot Loss vs Epoch
# -----------------------------
plt.figure()
plt.plot(losses)
plt.title("Loss vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# -----------------------------
# 7. Plot Accuracy vs Epoch
# -----------------------------
plt.figure()
plt.plot(accuracies)
plt.title("Accuracy vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()

# -----------------------------
# 8. Plot Sigmoid Function
# -----------------------------
x_vals = np.linspace(-10, 10, 100)
plt.figure()
plt.plot(x_vals, sigmoid(x_vals))
plt.title("Sigmoid Activation Function")
plt.xlabel("Input")
plt.ylabel("Output")
plt.show()

# -----------------------------
# 9. Decision Boundary
# -----------------------------
xx, yy = np.meshgrid(np.linspace(-0.5,1.5,200),
                     np.linspace(-0.5,1.5,200))

grid = np.c_[xx.ravel(), yy.ravel()]
z1 = np.dot(grid, W1) + b1
a1 = sigmoid(z1)
z2 = np.dot(a1, W2) + b2
preds = sigmoid(z2)
preds = preds.reshape(xx.shape)

plt.figure()
plt.contourf(xx, yy, preds > 0.5, alpha=0.5)
plt.scatter(X[:,0], X[:,1], c=y.flatten(), s=100)
plt.title("MLP Decision Boundary (XOR)")
plt.show()
