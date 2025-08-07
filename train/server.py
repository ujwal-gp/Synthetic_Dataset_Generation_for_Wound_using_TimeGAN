# server.py
import flwr as fl
import pickle
import os
import numpy as np
import xgboost as xgb

N_ROUNDS = 10
N_CLIENTS = 2
GLOBAL_MODEL_PATH = "global_model.pkl"

# ---- 1. Metric Aggregation Function ----
# Initialize global model once (dummy)
def initialize_global_model():
    model = xgb.XGBClassifier(n_estimators=1, max_depth=5, learning_rate=0.1)
    dummy_X = np.array([[0, 0], [1, 1]])
    dummy_y = np.array([0, 1])
    model.fit(dummy_X, dummy_y)
    with open(GLOBAL_MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

class Strategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        print(f"🔁 Aggregating round {server_round} with {len(results)} results...")

        if not results:
            return None, {}

        # Save the first client's model as the new global model (simple strategy)
        _, fit_res = results[0]
        parameters = fit_res.parameters

        # Deserialize model from client parameters
        model_bytes = parameters.tensors[0]
        with open(GLOBAL_MODEL_PATH, "wb") as f:
            f.write(model_bytes)

        print("✅ Global model updated from one of the clients.")

        return parameters, {}

def main():
    if not os.path.exists(GLOBAL_MODEL_PATH):
        initialize_global_model()

    strategy = Strategy(
        fraction_fit=1.0,             # All clients participate each round
        min_fit_clients=N_CLIENTS,
        min_available_clients=N_CLIENTS,
    )
    # Configure and start Flower server
    fl.server.start_server(
        server_address="172.16.174.136:8081",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=N_ROUNDS)
    )

if __name__ == "__main__":
    main()
