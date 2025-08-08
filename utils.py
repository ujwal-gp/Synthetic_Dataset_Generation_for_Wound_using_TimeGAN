# utils.py
import pandas as pd
import numpy as np
import os
import re
from sklearn.model_selection import train_test_split

def generate_labels(df):
    if "mqtt.clientid_len" not in df.columns and "mqtt.clientid" in df.columns:
        df["mqtt.clientid_len"] = df["mqtt.clientid"].astype(str).apply(len)
    # Threshold for oversized Client ID
    MAX_CLIENT_ID_LEN = 23
    oversized_client_id = df["mqtt.clientid_len"] > MAX_CLIENT_ID_LEN

    # Detect if topic is unusual
    def is_mqtt_topic_valid(topic: str) -> bool:
        try:
            # Handle missing or NaN topics early
            if pd.isna(topic) or str(topic).strip() == "":
                return False

            topic.encode('utf-8')

            if re.search(r'[+#]', topic):
                return False

            if re.search(r'\x00', topic):
                return False

            if re.match(r'^\$SYS/', topic):
                return False

            if re.search(r'//', topic):
                return False

            if re.search(r'/$', topic):
                return False

            if re.search(r'[^a-zA-Z0-9/_-]', topic):
                return False

            return True
        except Exception as e:
            print(f"❌ Unexpected error with topic '{topic}': {e}")
            return False

    # Apply the topic anomaly rule
    unusual_topic = df["mqtt.topic"].apply(lambda topic: not is_mqtt_topic_valid(topic))

    # Combine anomalies
    df["label"] = (oversized_client_id | unusual_topic).astype(int)

    return df

def load_partitioned_data(file_path, n_clients, test_size, output_dir):
    df = pd.read_csv(file_path)

    # Inject anomaly labels
    #df = generate_labels(df)
    print(f"✅ Total rows after labeling: {len(df)}")
    # Shuffle and partition
    split_dfs = np.array_split(df.sample(frac=1, random_state=42), n_clients)
    partitions = []

    os.makedirs(output_dir, exist_ok=True)
    for i, part in enumerate(split_dfs):
        if part.empty:
            print(f"⚠️ Skipping empty partition for client {i}")
            continue
        X = part.drop("label", axis=1)
        y = part["label"]

        if len(part) < 2:
            print(f"⚠️ Not enough data for train/test split for client {i}")
            continue
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)

        train_df.to_csv(os.path.join(output_dir, f"client_{i}_train.csv"), index=False)
        test_df.to_csv(os.path.join(output_dir, f"client_{i}_test.csv"), index=False)

        partitions.append({
            "train": (X_train, y_train),
            "test": (X_test, y_test)
        })
    return partitions
