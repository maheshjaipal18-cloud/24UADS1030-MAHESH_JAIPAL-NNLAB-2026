# # utils/feature_engineering.py

# import numpy as np

# def feature_engineering(df):
#     df['acc_mag'] = np.sqrt(df['X_Acc']**2 + df['Y_Acc']**2 + df['Z_Acc']**2)
#     df['gyro_mag'] = np.sqrt(df['X_Gyro']**2 + df['Y_Gyro']**2 + df['Z_Gyro']**2)

#     df['jerk'] = df.groupby('session_id')['acc_mag'].diff().fillna(0)

#     df['acc_mean'] = df.groupby('session_id')['acc_mag']\
#         .rolling(5).mean().reset_index(0, drop=True)

#     df['acc_std'] = df.groupby('session_id')['acc_mag']\
#         .rolling(5).std().reset_index(0, drop=True)

#     df = df.fillna(0)

#     return df

#new code 
import numpy as np

def feature_engineering(df):

    # =========================
    # 1. Magnitude Features
    # =========================
    df['acc_mag'] = np.sqrt(df['X_Acc']**2 + df['Y_Acc']**2 + df['Z_Acc']**2)
    df['gyro_mag'] = np.sqrt(df['X_Gyro']**2 + df['Y_Gyro']**2 + df['Z_Gyro']**2)

    # =========================
    # 2. Jerk (sudden change)
    # =========================
    df['jerk'] = df.groupby('session_id')['acc_mag'].diff().fillna(0)
    df['gyro_jerk'] = df.groupby('session_id')['gyro_mag'].diff().fillna(0)

    # =========================
    # 3. Direction Change
    # =========================
    df['acc_diff_x'] = df.groupby('session_id')['X_Acc'].diff().fillna(0)
    df['acc_diff_y'] = df.groupby('session_id')['Y_Acc'].diff().fillna(0)
    df['acc_diff_z'] = df.groupby('session_id')['Z_Acc'].diff().fillna(0)

    # =========================
    # 4. Rolling Statistics (VERY IMPORTANT)
    # =========================
    window = 10

    df['acc_mean'] = df.groupby('session_id')['acc_mag']\
        .rolling(window).mean().reset_index(0, drop=True)

    df['acc_std'] = df.groupby('session_id')['acc_mag']\
        .rolling(window).std().reset_index(0, drop=True)

    df['gyro_mean'] = df.groupby('session_id')['gyro_mag']\
        .rolling(window).mean().reset_index(0, drop=True)

    df['gyro_std'] = df.groupby('session_id')['gyro_mag']\
        .rolling(window).std().reset_index(0, drop=True)

    # =========================
    # 5. Peak Detection (Aggressive driving)
    # =========================
    df['high_acc_flag'] = (df['acc_mag'] > df['acc_mag'].quantile(0.90)).astype(int)
    df['high_gyro_flag'] = (df['gyro_mag'] > df['gyro_mag'].quantile(0.90)).astype(int)

    # =========================
    # 6. Energy Features
    # =========================
    df['acc_energy'] = df['acc_mag'] ** 2
    df['gyro_energy'] = df['gyro_mag'] ** 2

    # =========================
    # 7. Smoothness Feature
    # =========================
    df['smoothness'] = 1 / (1 + np.abs(df['jerk']))

    # =========================
    # Clean NaN
    # =========================
    df = df.fillna(0)

    return df