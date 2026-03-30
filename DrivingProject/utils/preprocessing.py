# utils/preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    df = df.drop(columns=['ID', 'SrNo'], errors='ignore')

    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values(['session_id', 'Timestamp'])

    df = df.ffill().bfill()

    # Remove outliers
    for col in ['X_Acc','Y_Acc','Z_Acc','X_Gyro','Y_Gyro','Z_Gyro']:
        df[col] = np.clip(df[col],
                          df[col].quantile(0.01),
                          df[col].quantile(0.99))

    return df


def normalize_data(df, scaler=None, train=True):
    cols = ['X_Acc','Y_Acc','Z_Acc','X_Gyro','Y_Gyro','Z_Gyro']

    if train:
        scaler = StandardScaler()
        df[cols] = scaler.fit_transform(df[cols])
    else:
        df[cols] = scaler.transform(df[cols])

    return df, scaler