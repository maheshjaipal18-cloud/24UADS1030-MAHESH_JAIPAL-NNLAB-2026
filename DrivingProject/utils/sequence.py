# utils/sequence.py

import numpy as np
from config import FEATURE_COLUMNS

def create_sequences(df, window_size, step):
    X = []
    y = []

    for session_id, group in df.groupby('session_id'):
        data = group[FEATURE_COLUMNS].values
        label = group['label'].iloc[0]

        for i in range(0, len(data) - window_size, step):
            X.append(data[i:i+window_size])
            y.append(label - 1)

    return np.array(X), np.array(y)