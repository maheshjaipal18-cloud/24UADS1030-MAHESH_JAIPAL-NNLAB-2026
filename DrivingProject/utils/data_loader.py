# utils/data_loader.py

import os
import pandas as pd

def load_data(data_dir):
    all_data = []
    session_id = 0

    for label in sorted(os.listdir(data_dir)):
        label_path = os.path.join(data_dir, label)

        if not os.path.isdir(label_path):
            continue

        for file in os.listdir(label_path):
            if file.endswith(".csv"):
                file_path = os.path.join(label_path, file)

                df = pd.read_csv(file_path)
                df['label'] = int(label.split('_')[0])
                df['session_id'] = session_id

                session_id += 1
                all_data.append(df)

    return pd.concat(all_data, ignore_index=True)