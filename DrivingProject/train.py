# train.py

import os
import joblib
from sklearn.model_selection import train_test_split

from utils.data_loader import load_data
from utils.preprocessing import preprocess_data, normalize_data
from utils.feature_engineering import feature_engineering
from utils.sequence import create_sequences
from utils.models import build_lstm, build_gru, build_transformer
from utils.evaluation import evaluate
import config

# Load data
df = load_data(config.DATA_DIR)

# Preprocess
df = preprocess_data(df)
df, scaler = normalize_data(df, train=True)
df = feature_engineering(df)

# Save scaler
joblib.dump(scaler, "models/scaler.pkl")

# Create sequences
X, y = create_sequences(df, config.WINDOW_SIZE, config.STEP_SIZE)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

input_shape = (X.shape[1], X.shape[2])

# Build models
models = {
    "lstm": build_lstm(input_shape),
    "gru": build_gru(input_shape),
    "transformer": build_transformer(input_shape)
}

# Train & evaluate
for name, model in models.items():
    print(f"\nTraining {name} model...")

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        X_train, y_train,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        validation_split=0.2
    )

    evaluate(model, X_test, y_test)

    model.save(f"models/{name}.h5")