# utils.py
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

FEATURE_COLS = [
    "Ammonia (ppm)", "Acetone (ppm)", "Ethanol (ppm)", "Hydrogen Sulfide (ppm)",
    "QoS", "message_length", "retained", "duplicate",
]

def to_int_safe(series, min_val=None, max_val=None, fill=0):
    s = pd.to_numeric(series, errors="coerce")
    if min_val is not None or max_val is not None:
        s = s.clip(lower=min_val if min_val is not None else s.min(),
                   upper=max_val if max_val is not None else s.max())
    s = s.round()
    if fill is not None:
        s = s.fillna(fill)
    return s.astype(int)

def to_bool_safe(series):
    if series.dtype == bool:
        return series
    s_num = pd.to_numeric(series, errors="coerce")
    if s_num.notna().any():
        return (s_num.fillna(0) >= 0.5)
    s_str = series.astype(str).str.strip().str.lower()
    return s_str.isin(["true","1","yes","y","t"])

def load_partitioned_data(file_path, n_clients, test_size, output_dir):
    df = pd.read_csv(file_path)

    if "label" not in df.columns:
        raise ValueError("Input CSV must contain a 'label' column.")

    # Ensure MQTT fields are numeric so client 'select_dtypes' includes them
    if "QoS" in df.columns:
        df["QoS"] = to_int_safe(df["QoS"], min_val=0, max_val=2, fill=0)
    if "message_length" in df.columns:
        df["message_length"] = to_int_safe(df["message_length"], min_val=0, fill=0)
    for b in ("retained", "duplicate"):
        if b in df.columns:
            # store as 0/1 ints for consistency (client scaling will include them)
            df[b] = to_bool_safe(df[b]).astype(int)

    print(f"✅ Total rows after labeling: {len(df)}")

    # Shuffle
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Split into n roughly equal partitions but try to stratify by label
    # Simple approach: split indices by class then interleave
    idx0 = df.index[df["label"] == 0].tolist()
    idx1 = df.index[df["label"] == 1].tolist()
    parts = [[] for _ in range(n_clients)]
    for k, idx_list in enumerate([idx0, idx1]):
        for i, idx in enumerate(idx_list):
            parts[i % n_clients].append(idx)

    os.makedirs(output_dir, exist_ok=True)
    partitions = []

    for i in range(n_clients):
        part_df = df.loc[parts[i]].sample(frac=1.0, random_state=42).reset_index(drop=True)
        if part_df.empty:
            print(f"⚠️ Skipping empty partition for client {i}")
            continue

        # Train/test split per client, stratified by label
        X = part_df.drop(columns=["label"])
        y = part_df["label"]
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
        except ValueError:
            # If a client has only one class, fall back to non-stratified split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)

        train_df.to_csv(os.path.join(output_dir, f"client_{i}_train.csv"), index=False)
        test_df.to_csv(os.path.join(output_dir, f"client_{i}_test.csv"), index=False)

        partitions.append({
            "train": (X_train, y_train),
            "test": (X_test, y_test)
        })

    return partitions
