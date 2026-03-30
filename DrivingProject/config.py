# config.py

DATA_DIR = "data/"
MODEL_DIR = "models/"

WINDOW_SIZE = 50
STEP_SIZE = 25

EPOCHS = 20
BATCH_SIZE = 32

FEATURE_COLUMNS = [
    'X_Acc','Y_Acc','Z_Acc',
    'X_Gyro','Y_Gyro','Z_Gyro',
    'acc_mag','gyro_mag','jerk',
    'acc_mean','acc_std'
]

FEATURE_COLUMNS = [
    'X_Acc','Y_Acc','Z_Acc',
    'X_Gyro','Y_Gyro','Z_Gyro',
    'acc_mag','gyro_mag','jerk','gyro_jerk',
    'acc_diff_x','acc_diff_y','acc_diff_z',
    'acc_mean','acc_std','gyro_mean','gyro_std',
    'high_acc_flag','high_gyro_flag',
    'acc_energy','gyro_energy','smoothness'
]