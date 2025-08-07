# server.py

import flwr as fl
import numpy as np
import os
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

# Settings
N_ROUNDS = 10
N_CLIENTS = 2
GLOBAL_MODEL_PATH = "global_model.json"

# ---- 1. Initialize a Dummy XGBoost Model (once) ----
import xgboost as xgb

def initialize_global_model():
    model = xgb.XGBClassifier(n_estimators=1, max_depth=2)
    X_dummy = np.array([[0, 0], [1, 1]])
    y_dummy = np.array([0, 1])
    model.fit(X_dummy, y_dummy)

    model.get_booster().save_model(GLOBAL_MODEL_PATH)
    print("📦 Initial global model saved.")


# ---- 2. Custom Federated Averaging Strategy ----
class Strategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        print(f"🔁 Aggregating Round {server_round} from {len(results)} clients...")

        if not results:
            print("⚠️ No results to aggregate.")
            return None, {}
        
        params = results[0][1].parameters

        # ✅ Convert Flower Parameters -> list of np.ndarrays
        arrays = parameters_to_ndarrays(params)
        # You sent exactly one array (uint8 buffer of the JSON string)
        buf = arrays[0]                       # dtype=uint8, shape=(N,)
        model_json = buf.tobytes().decode("utf-8")
        with open("global_model.json", "w", encoding="utf-8") as f:
            f.write(model_json)

        return params, {}

    def initialize_parameters(self, client_manager):
        if os.path.exists(GLOBAL_MODEL_PATH):
            with open("global_model.json", "r", encoding="utf-8") as f:
                json_str = f.read()
            return ndarrays_to_parameters([np.frombuffer(json_str.encode("utf-8"), dtype=np.uint8)])
        else:
            print("⚠️ No global model found — will initialize from client.")
            return None


# ---- 3. Main Server Loop ----
def main():
    if not os.path.exists(GLOBAL_MODEL_PATH):
        print("🔧 Initializing global model...")
        initialize_global_model()

    strategy = Strategy(
        fraction_fit=1.0,
        min_fit_clients=N_CLIENTS,
        min_available_clients=N_CLIENTS,
    )

    print("🚀 Starting Flower server...")
    fl.server.start_server(
        server_address="172.16.174.136:8081",
        config=fl.server.ServerConfig(num_rounds=N_ROUNDS),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
