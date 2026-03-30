# utils/evaluation.py

from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    preds = np.argmax(preds, axis=1)

    print("Classification Report:")
    print(classification_report(y_test, preds))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))