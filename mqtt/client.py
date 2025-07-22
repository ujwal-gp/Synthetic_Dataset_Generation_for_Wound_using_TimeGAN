# client.py
import flwr as fl
import xgboost as xgb
import pandas as pd
import numpy as np

class XGBClient(fl.client.NumPyClient):
    def __init__(self, model, X, y):
        self.model = model
        self.X = X
        self.y = y

    def get_parameters(self):
        # Return model weights (for XGBoost, using dictionary of feature scores as example)
        return [param.tolist() for param in self.model.get_booster().get_dump()]

    def fit(self, parameters, config):
        # For simplicity, reinitialize model then train on local data
        self.model = xgb.XGBClassifier(objective="binary:logistic")
        self.model.fit(self.X, self.y)
        # Return updated weights (dump) and number of samples
        return self.model.get_booster().get_dump(), len(self.X)

    def evaluate(self, parameters, config):
        loss = 0.0
        accuracy = 0.0
        # (Evaluation logic could be added here)
        return float(loss), len(self.X), {"accuracy": accuracy}

if __name__ == "__main__":
    # Load this client's data partition
    df = pd.read_csv("client1_data.csv")  # or client2_data.csv
    X = df.drop(columns=["Arrhythmia_Event", "Signal_Dropout"]).values
    # For example, predict arrhythmia event
    y = df["Arrhythmia_Event"].values
    # Initialize XGBoost model
    model = xgb.XGBClassifier(objective="binary:logistic", use_label_encoder=False)
    client = XGBClient(model, X, y)
    fl.client.start_client(server_address="localhost:8080", client=client)
