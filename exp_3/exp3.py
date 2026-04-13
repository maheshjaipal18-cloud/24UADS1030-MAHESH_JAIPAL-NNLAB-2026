import tensorflow as tf
import numpy as np

# Disable eager for graph-style (optional but closer to classic TF)
tf.compat.v1.disable_eager_execution()

# Load MNIST Dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize
x_train = x_train.reshape(-1, 784) / 255.0
x_test  = x_test.reshape(-1, 784) / 255.0

# One-hot encoding
y_train = np.eye(10)[y_train]
y_test  = np.eye(10)[y_test]

# Placeholders
X = tf.compat.v1.placeholder(tf.float32, [None, 784])
Y = tf.compat.v1.placeholder(tf.float32, [None, 10])

# Weights and Biases
W1 = tf.Variable(tf.random.normal([784, 128]))
b1 = tf.Variable(tf.zeros([128]))

W2 = tf.Variable(tf.random.normal([128, 10]))
b2 = tf.Variable(tf.zeros([10]))

# Feed Forward
hidden_layer = tf.nn.relu(tf.matmul(X, W1) + b1)
output_layer = tf.matmul(hidden_layer, W2) + b2

# Loss Function
loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(
        logits=output_layer, labels=Y
    )
)

# Backpropagation (Optimizer)
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001)
train_step = optimizer.minimize(loss)

# Accuracy
correct_pred = tf.equal(tf.argmax(output_layer, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Training
epochs = 10
batch_size = 128

init = tf.compat.v1.global_variables_initializer()

with tf.compat.v1.Session() as sess:
    sess.run(init)

    for epoch in range(epochs):
        for i in range(0, len(x_train), batch_size):
            batch_x = x_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]

            sess.run(train_step, feed_dict={X: batch_x, Y: batch_y})

        train_loss = sess.run(loss, feed_dict={X: batch_x, Y: batch_y})
        train_acc  = sess.run(accuracy, feed_dict={X: batch_x, Y: batch_y})

        print(f"Epoch {epoch+1}, Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

    # Testing
    test_acc = sess.run(accuracy, feed_dict={X: x_test, Y: y_test})
    print("\nTest Accuracy:", test_acc)