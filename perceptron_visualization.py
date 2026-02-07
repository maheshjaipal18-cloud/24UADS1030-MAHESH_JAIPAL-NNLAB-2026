import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

class Perceptron:
    def __init__(self, lr=0.1, epochs=20):
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        self.b = 0
        self.errors = []

        for _ in range(self.epochs):
            error_count = 0
            for xi, target in zip(X, y):
                y_pred = self.predict(xi)
                update = self.lr * (target - y_pred)
                self.w += update * xi
                self.b += update
                error_count += int(update != 0)
            self.errors.append(error_count)

    def net_input(self, X):
        return np.dot(X, self.w) + self.b

    def predict(self, X):
        return np.where(self.net_input(X) >= 0, 1, 0)

def plot_decision_boundary(X, y, model, title):
    x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
    y_min, y_max = X[:,1].min()-1, X[:,1].max()+1

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:,0], X[:,1], c=y, edgecolor='k', s=100)
    plt.title(title)
    plt.xlabel("Input X1")
    plt.ylabel("Input X2")
    plt.show()

# ---------- NAND ----------
X_nand = np.array([[0,0],[0,1],[1,0],[1,1]])
y_nand = np.array([1,1,1,0])

p_nand = Perceptron()
p_nand.fit(X_nand, y_nand)

plot_decision_boundary(X_nand, y_nand, p_nand,
                       "Perceptron for NAND")

plt.plot(p_nand.errors, marker='o')
plt.title("Training Errors - NAND")
plt.xlabel("Epochs")
plt.ylabel("Errors")
plt.show()

# ---------- XOR ----------
X_xor = np.array([[0,0],[0,1],[1,0],[1,1]])
y_xor = np.array([0,1,1,0])

p_xor = Perceptron()
p_xor.fit(X_xor, y_xor)

plot_decision_boundary(X_xor, y_xor, p_xor,
                       "Perceptron for XOR")

plt.plot(p_xor.errors, marker='o')
plt.title("Training Errors - XOR")
plt.xlabel("Epochs")
plt.ylabel("Errors")
plt.show()
