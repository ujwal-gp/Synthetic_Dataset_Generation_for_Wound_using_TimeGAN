# client.py
import flwr as fl
import xgboost as xgb
import numpy as np
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler
import pickle
import sys

# Load client-specific data
def load_client_data(client_id, data_dir="../client_data"):
    train_file = os.path.join(data_dir, f"client_{client_id}_train.csv")
    test_file = os.path.join(data_dir, f"client_{client_id}_test.csv")
    print(f'test file {train_file}')
    print(f'test file {test_file}')

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    # Split into features and labels
    X_train = train_df.drop("label", axis=1)
    y_train = train_df["label"]
    X_test = test_df.drop("label", axis=1)
    y_test = test_df["label"]

    # Filter numeric columns only
    X_train_numeric = X_train.select_dtypes(include=[np.number]).fillna(0)
    X_test_numeric = X_test.select_dtypes(include=[np.number]).fillna(0)

    # Scale features
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_numeric)
    X_test_scaled = scaler.transform(X_test_numeric)

    return X_train_scaled, y_train.values, X_test_scaled, y_test.values

class XGBClient(fl.client.NumPyClient):
    def __init__(self, X_train, y_train, X_test, y_test, client_id):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.client_id = client_id
        self.model_path = f"xgb_model_client{client_id}.pkl"

        if os.path.exists(self.model_path):
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
            print(f"✅ Loaded model from {self.model_path}")
        else:
            self.model = xgb.XGBClassifier()
            print(f"🆕 Initialized new model for Client {client_id}")
        self.fitted = False  # ✅ Initialize fitted flag

    def get_parameters(self, config):
        if self.fitted:
            raw_bytes = self.model.get_booster().save_raw()
            return [np.frombuffer(raw_bytes, dtype=np.uint8)]
        else:
            return [np.zeros(1, dtype=np.uint8)]

    def fit(self, parameters, config):
        self.model.fit(self.X_train, self.y_train)
        self.fitted = True  # ✅ Set flag after training
        return self.get_parameters(config), len(self.X_train), {}

    def evaluate(self, parameters, config):
        if parameters:
            model_bytes = parameters[0]
            self.model.load_model(bytearray(model_bytes))
        preds = self.model.predict(self.X_test)

        accuracy = accuracy_score(self.y_test, preds)
        precision = precision_score(self.y_test, preds, average='weighted', zero_division=0)
        recall = recall_score(self.y_test, preds, average='weighted', zero_division=0)
        f1 = f1_score(self.y_test, preds, average='weighted', zero_division=0)

        print(f"📊 Client {self.client_id} Evaluation - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

        return float(1 - accuracy), len(self.X_test), {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1)
        }

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python client.py <client_id>")
        sys.exit(1)

    client_id = int(sys.argv[1])
    X_train, y_train, X_test, y_test = load_client_data(client_id)
    fl.client.start_client(server_address="172.16.174.136:8081", client=XGBClient(X_train, y_train, X_test, y_test, client_id).to_client())
