# predict.py

import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

from utils.preprocessing import preprocess_data, normalize_data
from utils.feature_engineering import feature_engineering
from utils.sequence import create_sequences
import config

def interpret(score):
    return {
        1: "Dangerous",
        2: "Rash",
        3: "Normal",
        4: "Good",
        5: "Excellent"
    }[score]


def predict(file_path):
    scaler = joblib.load("models/scaler.pkl")

    models = [
        load_model("models/lstm.h5"),
        load_model("models/gru.h5"),
        load_model("models/transformer.h5")
    ]

    df = pd.read_csv(file_path)
    df['session_id'] = 0
    df['label'] = 3  # dummy

    df = preprocess_data(df)
    df, _ = normalize_data(df, scaler, train=False)
    df = feature_engineering(df)

    X, _ = create_sequences(df, config.WINDOW_SIZE, config.STEP_SIZE)

    preds = []
    for model in models:
        p = np.argmax(model.predict(X), axis=1)
        preds.append(np.mean(p))

    final_score = int(round(np.mean(preds))) + 1

    print("Final Driving Score:", final_score)
    print("Behavior:", interpret(final_score))


# Example usage
if __name__ == "__main__":
    import sys

if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = "data/5_star/sonu_verysafe02.csv"   # default path

    predict(file_path)