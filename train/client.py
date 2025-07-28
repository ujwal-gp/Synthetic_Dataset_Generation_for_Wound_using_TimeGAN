# client.py
import flwr as fl
import xgboost as xgb
import numpy as np
from utils import load_partitioned_data

class XGBClient(fl.client.NumPyClient):
    def __init__(self, X, y):
        self.model = xgb.XGBClassifier()
        self.X, self.y = X, y

    def get_parameters(self, config):
        return []  # XGBoost is tree-based; weights not passed directly

    def fit(self, parameters, config):
        self.model.fit(self.X, self.y)
        return [], len(self.X), {}

    def evaluate(self, parameters, config):
        preds = self.model.predict(self.X)
        acc = np.mean(preds == self.y)
        return float(1.0 - acc), len(self.X), {"accuracy": acc}

if __name__ == "__main__":
    import sys
    client_id = int(sys.argv[1])
    partitions = load_partitioned_data()
    X, y = partitions[client_id]
    fl.client.start_client(server_address="localhost:8081", client=XGBClient(X, y).to_client())
