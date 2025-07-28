# utils.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_partitioned_data(file_path="../synthetic_wound_gas_flat.csv", n_clients=2):
    df = pd.read_csv(file_path)

    # Inject binary labels (example: random anomalies)
    df["label"] = (df["O2"] < 0.4).astype(int)  # simple rule for anomaly

    partitions = []
    split_dfs = np.array_split(df.sample(frac=1, random_state=42), n_clients)
    for part in split_dfs:
        X = part.drop("label", axis=1)
        y = part["label"]
        partitions.append((X, y))
    return partitions
