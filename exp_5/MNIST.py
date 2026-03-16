import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

# ===============================
# Load Dataset
# ===============================
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

# Reduce dataset for faster execution
x_train = x_train[:15000]
y_train = y_train[:15000]

x_test = x_test[:3000]
y_test = y_test[:3000]

# Normalize
x_train = x_train/255.0
x_test = x_test/255.0

# Reshape
x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)

# ===============================
# CNN Model Function
# ===============================
def create_model(filter_size=3, reg=None, optimizer='adam'):

    model = keras.Sequential([

        layers.Conv2D(32,(filter_size,filter_size),
                      activation='relu',
                      kernel_regularizer=reg,
                      input_shape=(28,28,1)),

        layers.MaxPooling2D((2,2)),

        layers.Conv2D(64,(filter_size,filter_size),
                      activation='relu',
                      kernel_regularizer=reg),

        layers.MaxPooling2D((2,2)),

        layers.Flatten(),

        layers.Dense(128,activation='relu'),

        layers.Dense(10,activation='softmax')
    ])

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# ===============================
# Function to Show Results
# ===============================
def show_results(history, model, title):

    # Accuracy Graph
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(title + " Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["Train","Validation"])
    plt.show()

    # Loss Graph
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(title + " Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["Train","Validation"])
    plt.show()

    # Confusion Matrix
    predictions = model.predict(x_test)
    y_pred = np.argmax(predictions,axis=1)

    cm = confusion_matrix(y_test,y_pred)

    plt.figure(figsize=(6,5))
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues')
    plt.title(title + " Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# ===============================
# 1 Filter Size Variation
# ===============================
print("\n FILTER SIZE VARIATION")

for f in [3,5,7]:

    print("\nTraining Filter Size:",f)

    model = create_model(filter_size=f)

    history = model.fit(
        x_train,y_train,
        epochs=3,
        batch_size=64,
        validation_data=(x_test,y_test)
    )

    loss,acc = model.evaluate(x_test,y_test,verbose=0)
    print("Test Accuracy:",acc)

    show_results(history,model,"Filter Size "+str(f))

# ===============================
# 2 Regularization Variation
# ===============================
print("\n REGULARIZATION VARIATION")

regularizations = {
    "None":None,
    "L1":regularizers.l1(0.001),
    "L2":regularizers.l2(0.001)
}

for name,reg in regularizations.items():

    print("\nTraining Regularization:",name)

    model = create_model(reg=reg)

    history = model.fit(
        x_train,y_train,
        epochs=3,
        batch_size=64,
        validation_data=(x_test,y_test)
    )

    loss,acc = model.evaluate(x_test,y_test,verbose=0)
    print("Test Accuracy:",acc)

    show_results(history,model,"Regularization "+name)

# ===============================
# 3 Batch Size Variation
# ===============================
print("\n BATCH SIZE VARIATION")

for b in [32,64,128]:

    print("\nTraining Batch Size:",b)

    model = create_model()

    history = model.fit(
        x_train,y_train,
        epochs=3,
        batch_size=b,
        validation_data=(x_test,y_test)
    )

    loss,acc = model.evaluate(x_test,y_test,verbose=0)
    print("Test Accuracy:",acc)

    show_results(history,model,"Batch Size "+str(b))

# ===============================
# 4 Optimizer Variation
# ===============================
print("\n OPTIMIZER VARIATION")

for opt in ['adam','sgd','rmsprop']:

    print("\nTraining Optimizer:",opt)

    model = create_model(optimizer=opt)

    history = model.fit(
        x_train,y_train,
        epochs=3,
        batch_size=64,
        validation_data=(x_test,y_test)
    )

    loss,acc = model.evaluate(x_test,y_test,verbose=0)
    print("Test Accuracy:",acc)

    show_results(history,model,"Optimizer "+opt)
