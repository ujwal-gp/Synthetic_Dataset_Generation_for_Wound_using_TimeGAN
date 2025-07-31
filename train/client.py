# client.py
import flwr as fl
import xgboost as xgb
import numpy as np
import os
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils import load_partitioned_data

class XGBClient(fl.client.NumPyClient):
    def __init__(self, train_data, test_data, client_id):
        self.X_train, self.y_train = train_data
        self.X_test, self.y_test = test_data
        self.client_id = client_id

        self.model_path = f"xgb_model_client{client_id}.pkl"

        if os.path.exists(self.model_path):
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
            print(f"✅ Loaded model from {self.model_path}")
        else:
            self.model = xgb.XGBClassifier()
            print(f"🆕 Initialized new model for Client {client_id}")

    def get_parameters(self, config):
        return []

    def fit(self, parameters, config):
        self.model.fit(self.X_train, self.y_train)
        return self.get_parameters(config), len(self.X_train), {}

    def evaluate(self, parameters, config):
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
    import sys
    client_id = int(sys.argv[1])
    partitions = load_partitioned_data()
    train_data = partitions[client_id]["train"]
    test_data = partitions[client_id]["test"]
    fl.client.start_client(server_address="localhost:8081", client=XGBClient(train_data, test_data, client_id).to_client())
