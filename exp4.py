import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# ================= LOAD MNIST =================
def load_mnist(batch_size):
    ds_train, ds_test = tfds.load('mnist', split=['train', 'test'], as_supervised=True)

    def preprocess(img, lbl):
        img = tf.reshape(tf.cast(img, tf.float32)/255.0, (784,))
        return img, tf.cast(lbl, tf.int64)

    train_ds = ds_train.map(preprocess).shuffle(10000).batch(batch_size)
    test_ds = ds_test.map(preprocess).batch(batch_size)
    return train_ds, test_ds

# ================= MODEL CREATION =================
def create_weights(hidden_units):
    W1 = tf.Variable(tf.random.truncated_normal([784, hidden_units], stddev=0.1))
    b1 = tf.Variable(tf.zeros([hidden_units]))
    W2 = tf.Variable(tf.random.truncated_normal([hidden_units, 10], stddev=0.1))
    b2 = tf.Variable(tf.zeros([10]))
    return W1, b1, W2, b2

# ================= FORWARD PASS =================
def forward(x, W1, b1, W2, b2, activation):
    z1 = tf.matmul(x, W1) + b1
    if activation=="relu": h1 = tf.nn.relu(z1)
    elif activation=="sigmoid": h1 = tf.nn.sigmoid(z1)
    else: h1 = tf.nn.tanh(z1)
    logits = tf.matmul(h1, W2) + b2
    return logits

# ================= LOSS & ACCURACY =================
def compute_loss(logits, labels):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))

def compute_acc(logits, labels):
    preds = tf.argmax(logits, axis=1)
    return tf.reduce_mean(tf.cast(tf.equal(preds, labels), tf.float32))

# ================= TRAINING =================
def train_model(activation="relu", hidden_units=128, lr=0.1, batch_size=64, epochs=10):
    train_ds, test_ds = load_mnist(batch_size)
    W1, b1, W2, b2 = create_weights(hidden_units)
    optimizer = tf.optimizers.SGD(lr)
    history = {"acc":[]}

    print(f"\nTraining: Activation={activation}, Hidden={hidden_units}, LR={lr}, Batch={batch_size}, Epochs={epochs}")

    for epoch in range(epochs):
        epoch_acc, batches = 0, 0
        for x_batch, y_batch in train_ds:
            with tf.GradientTape() as tape:
                logits = forward(x_batch, W1, b1, W2, b2, activation)
                loss = compute_loss(logits, y_batch)
            grads = tape.gradient(loss, [W1,b1,W2,b2])
            optimizer.apply_gradients(zip(grads, [W1,b1,W2,b2]))
            epoch_acc += compute_acc(logits, y_batch).numpy()
            batches += 1
        history["acc"].append(epoch_acc/batches)
        print(f"Epoch {epoch+1:02d} | Acc={history['acc'][-1]:.4f}")

    return history, W1, b1, W2, b2, test_ds

# ================= CONFUSION MATRIX =================
def draw_cm(W1,b1,W2,b2,activation,test_ds,title):
    all_preds, all_labels = [], []
    for x_batch, y_batch in test_ds:
        logits = forward(x_batch, W1, b1, W2, b2, activation)
        preds = tf.argmax(logits, axis=1)
        all_preds.extend(preds.numpy())
        all_labels.extend(y_batch.numpy())
    cm = tf.math.confusion_matrix(all_labels, all_preds).numpy()
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for i in range(10):
        for j in range(10):
            plt.text(j, i, cm[i,j], ha="center", va="center", color="white" if cm[i,j]>cm.max()/2 else "black")
    plt.colorbar()
    plt.show()

# ================= EXPERIMENTS =================

# 🔹 Activation Variation
activations = ["relu","sigmoid","tanh"]
for act in activations:
    history, W1, b1, W2, b2, test_ds = train_model(activation=act)
    plt.plot(history["acc"])
    plt.title(f"Accuracy - Activation={act}")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.grid(); plt.show()
    draw_cm(W1,b1,W2,b2,act,test_ds,f"Confusion Matrix - Activation={act}")

# 🔹 Batch Size Variation
batch_sizes = [64,128,256]
for bs in batch_sizes:
    history, W1, b1, W2, b2, test_ds = train_model(batch_size=bs)
    plt.plot(history["acc"])
    plt.title(f"Accuracy - Batch Size={bs}")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.grid(); plt.show()
    draw_cm(W1,b1,W2,b2,"relu",test_ds,f"Confusion Matrix - Batch={bs}")

# 🔹 Learning Rate Variation
learning_rates = [0.01,0.05,0.1]
for lr in learning_rates:
    history, W1, b1, W2, b2, test_ds = train_model(lr=lr)
    plt.plot(history["acc"])
    plt.title(f"Accuracy - Learning Rate={lr}")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.grid(); plt.show()
    draw_cm(W1,b1,W2,b2,"relu",test_ds,f"Confusion Matrix - LR={lr}")
