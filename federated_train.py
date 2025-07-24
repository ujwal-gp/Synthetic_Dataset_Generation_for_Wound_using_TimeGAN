import pandas as pd
from sklearn.model_selection import train_test_split

# Load and preprocess MQTTset data
mqtt_df = pd.read_csv("MQTTset.csv")  # assuming a preprocessed CSV with features and 'label'
# Select relevant features (example: client ID length and topic length) and the label
mqtt_df["clientID_len"] = mqtt_df["mqtt.clientid"].str.len()
mqtt_df["topic_len"] = mqtt_df["mqtt.topic"].str.len()
mqtt_features = mqtt_df[["clientID_len", "topic_len"]].values
mqtt_labels = mqtt_df["label"].values   # 0 for normal, 1 for attack/anomaly

# Load synthetic gas sensor data (already prepared as DataFrame)
gas_df = pd.read_csv("synthetic_wound_gas_flat.csv")  # columns: O2, NO, H2S, CO, label
gas_features = gas_df[["O2", "NO", "H2S", "CO"]].values
gas_labels = gas_df["label"].values  # 0 = normal, 1 = anomaly

# Pad feature vectors to have equal length (6 in this example)
import numpy as np
mqtt_padded = np.hstack([mqtt_features, np.zeros((len(mqtt_features), 4))])  # pad 4 zeros for gas features
gas_padded = np.hstack([np.zeros((len(gas_features), 2)), gas_features])    # pad 2 zeros for MQTT features

X = np.vstack([mqtt_padded, gas_padded])
y = np.concatenate([mqtt_labels, gas_labels])

# Split into training and global test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Partition training data among clients (e.g., 2 clients)
num_clients = 2
client_partitions = np.array_split(np.arange(len(X_train)), num_clients)
client_datasets = []
for indices in client_partitions:
    client_datasets.append((X_train[indices], y_train[indices]))
print(f"Prepared data for {num_clients} clients.")
