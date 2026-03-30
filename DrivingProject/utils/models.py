# # utils/models.py

# import tensorflow as tf
# from tensorflow.keras.models import Sequential, Model
# from tensorflow.keras.layers import (
#     LSTM, GRU, Dense, Dropout,
#     Input, LayerNormalization, MultiHeadAttention
# )

# # 🔵 LSTM MODEL
# def build_lstm(input_shape):
#     model = Sequential([
#         LSTM(64, return_sequences=True, input_shape=input_shape),
#         Dropout(0.3),
#         LSTM(32),
#         Dense(32, activation='relu'),
#         Dense(5, activation='softmax')
#     ])
#     return model


# # 🟢 GRU MODEL
# def build_gru(input_shape):
#     model = Sequential([
#         GRU(64, return_sequences=True, input_shape=input_shape),
#         Dropout(0.3),
#         GRU(32),
#         Dense(32, activation='relu'),
#         Dense(5, activation='softmax')
#     ])
#     return model


# # 🟣 TRANSFORMER MODEL
# def build_transformer(input_shape):
#     inputs = Input(shape=input_shape)

#     x = MultiHeadAttention(num_heads=2, key_dim=32)(inputs, inputs)
#     x = LayerNormalization()(x)

#     x = Dense(64, activation='relu')(x)
#     x = Dense(32, activation='relu')(x)

#     # ✅ FIXED LINE
#     x = GlobalAveragePooling1D()(x)

#     outputs = Dense(5, activation='softmax')(x)

#     return Model(inputs, outputs)

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, GRU, Dense, Dropout,
    Input, LayerNormalization, MultiHeadAttention,
    GlobalAveragePooling1D
)
from tensorflow.keras.layers import Bidirectional

def build_lstm(input_shape):
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape),
        Dropout(0.3),

        Bidirectional(LSTM(32)),
        Dense(64, activation='relu'),
        Dropout(0.3),

        Dense(5, activation='softmax')
    ])
    return model
    
def build_lstm(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(32),
        Dense(32, activation='relu'),
        Dense(5, activation='softmax')
    ])
    return model


def build_gru(input_shape):
    model = Sequential([
        GRU(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        GRU(32),
        Dense(32, activation='relu'),
        Dense(5, activation='softmax')
    ])
    return model


def build_transformer(input_shape):
    inputs = Input(shape=input_shape)

    x = MultiHeadAttention(num_heads=2, key_dim=32)(inputs, inputs)
    x = LayerNormalization()(x)

    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)

    x = GlobalAveragePooling1D()(x)  # ✅ now works

    outputs = Dense(5, activation='softmax')(x)

    return Model(inputs, outputs)

