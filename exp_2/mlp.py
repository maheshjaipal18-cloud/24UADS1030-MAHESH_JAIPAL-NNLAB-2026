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

def sigmoid_derivative(a):
    return a * (1 - a)

# Plot Sigmoid Function
x_vals = np.linspace(-10, 10, 100)
plt.figure()
plt.plot(x_vals, sigmoid(x_vals))
plt.title("Sigmoid Activation Function")
plt.xlabel("Input")
plt.ylabel("Output")
plt.grid()
plt.show()

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
# 4. Training
# -----------------------------
for epoch in range(epochs):

    # Forward Propagation
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)

    z2 = np.dot(a1, W2) + b2
    y_pred = sigmoid(z2)

    # Binary Cross Entropy Loss
    loss = -np.mean(y*np.log(y_pred+1e-8) + (1-y)*np.log(1-y_pred+1e-8))
    losses.append(loss)

    # Accuracy
    predictions = (y_pred > 0.5).astype(int)
    accuracy = np.mean(predictions == y)
    accuracies.append(accuracy)

    # Print at intervals
    if epoch % 1000 == 0:
        print(f"\nEpoch {epoch}")
        for i in range(len(X)):
            print(f"Input: {X[i]}, Target: {y[i][0]}, "
                  f"Predicted: {y_pred[i][0]:.4f}, "
                  f"Loss: {loss:.4f}")

    # Backpropagation
    dz2 = y_pred - y
    dW2 = np.dot(a1.T, dz2)
    db2 = np.sum(dz2, axis=0, keepdims=True)

    dz1 = np.dot(dz2, W2.T) * sigmoid_derivative(a1)
    dW1 = np.dot(X.T, dz1)
    db1 = np.sum(dz1, axis=0, keepdims=True)

    # Update weights
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

# -----------------------------
# 5. Final Evaluation
# -----------------------------
final_predictions = sigmoid(np.dot(sigmoid(np.dot(X, W1)+b1), W2)+b2)
rounded_predictions = (final_predictions > 0.5).astype(int)
training_accuracy = np.mean(rounded_predictions == y)

# Confusion Matrix
TP = np.sum((rounded_predictions == 1) & (y == 1))
TN = np.sum((rounded_predictions == 0) & (y == 0))
FP = np.sum((rounded_predictions == 1) & (y == 0))
FN = np.sum((rounded_predictions == 0) & (y == 1))

confusion_matrix = np.array([[TN, FP],
                             [FN, TP]])

print("\nFinal Predictions (Raw):")
print(final_predictions)

print("\nRounded Predictions:")
print(rounded_predictions)

print("\nTraining Accuracy:", training_accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix)

# -----------------------------
# 6. Plot Loss vs Epoch
# -----------------------------
plt.figure()
plt.plot(losses)
plt.title("Loss vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.show()

# -----------------------------
# 7. Plot Accuracy vs Epoch
# -----------------------------
plt.figure()
plt.plot(accuracies)
plt.title("Accuracy vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid()
plt.show()

# -----------------------------
# 8. Decision Boundary
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
plt.xlabel("Input 1")
plt.ylabel("Input 2")
plt.show()
