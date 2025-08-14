# client.py
import flwr as fl
import xgboost as xgb
import numpy as np
import os
import json
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import sys
from sklearn.exceptions import NotFittedError

# --- plotting (headless-safe) ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


FEATURE_COLS = [
    "Ammonia (ppm)", "Acetone (ppm)", "Ethanol (ppm)", "Hydrogen Sulfide (ppm)",
    "QoS", "message_length", "retained", "duplicate",
]

def load_client_data(client_id, data_dir="../client_data"):
    train_file = os.path.join(data_dir, f"client_{client_id}_train.csv")
    test_file = os.path.join(data_dir, f"client_{client_id}_test.csv")

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    # Enforce schema / order
    X_train = train_df[FEATURE_COLS].copy()
    y_train = train_df["label"].values
    X_test  = test_df[FEATURE_COLS].copy()
    y_test  = test_df["label"].values

    # Ensure bools are numeric
    for b in ("retained", "duplicate"):
        if b in X_train.columns:
            X_train[b] = X_train[b].astype(int)
            X_test[b]  = X_test[b].astype(int)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train.values.astype(float))
    X_test_scaled  = scaler.transform(X_test.values.astype(float))

    return X_train_scaled, y_train, X_test_scaled, y_test


class XGBClient(fl.client.NumPyClient):
    def __init__(self, X_train, y_train, X_test, y_test, client_id):
        self.client_id = client_id
        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test

        # Start with 1 tree so we can do a minimal cold-start fit if needed
        self.model = xgb.XGBClassifier(
            n_estimators=1,               # <- was 0; 1 enables a tiny warmup fit
            max_depth=6,
            learning_rate=0.3,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            eval_metric="logloss",
            n_jobs=-1,
        )

    def _ensure_fitted(self):
        """If booster does not exist yet (cold start), fit 1 round on local data."""
        try:
            # Will raise NotFittedError if not trained/loaded yet
            _ = self.model.get_booster()
        except NotFittedError:
            # Minimal fit to materialize a booster
            self.model.set_params(n_estimators=max(1, self.model.get_params().get("n_estimators", 1)))
            self.model.fit(self.X_train, self.y_train, verbose=False)

    def get_parameters(self, config):
        """Return model parameters; do a tiny warmup fit if this is the very first call."""
        self._ensure_fitted()
        model_path = f"client_model_{self.client_id}.json"
        self.model.save_model(model_path)  # JSON format
        with open(model_path, "r", encoding="utf-8") as f:
            model_json_str = f.read()
        return [np.frombuffer(model_json_str.encode("utf-8"), dtype=np.uint8)]

    def fit(self, parameters, config):
        # Load global model from server (if provided)
        xgb_model_arg = None
        if parameters and parameters[0].size > 1:
            json_str = bytes(parameters[0].tolist()).decode("utf-8")
            model_path = f"client_model_{self.client_id}.json"
            with open(model_path, "w", encoding="utf-8") as f:
                f.write(json_str)
            self.model.load_model(model_path)
            xgb_model_arg = model_path

        rounds_to_add = int(config.get("rounds_to_add", 25))
        if rounds_to_add <= 0:
            rounds_to_add = 25

        # Continue boosting from provided global model if available
        current_estimators = self.model.get_params().get("n_estimators", 1)
        self.model.set_params(n_estimators=current_estimators + rounds_to_add)

        if xgb_model_arg:
            self.model.fit(self.X_train, self.y_train, xgb_model=xgb_model_arg, verbose=False)
        else:
            # First round on this client (cold start already handled, but train more now)
            self.model.fit(self.X_train, self.y_train, verbose=False)

        # Local eval for server selection + FI vector
        preds = self.model.predict(self.X_test)
        acc = accuracy_score(self.y_test, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(self.y_test, preds, average="binary", zero_division=0)

        metrics = {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
        }

        # Feature importance vector aligned with array order
        try:
            booster = self.model.get_booster()
            fi = booster.get_score(importance_type="gain")
            vec = np.zeros(self.X_train.shape[1], dtype=float)
            for k, v in fi.items():
                if k.startswith("f"):
                    idx = int(k[1:])
                    if 0 <= idx < len(vec):
                        vec[idx] = float(v)
            metrics["feature_importance_json"] = json.dumps(vec.tolist())
        except Exception:
            pass

        return self.get_parameters(config), len(self.X_train), metrics

    def evaluate(self, parameters, config):
        # Evaluate the provided global model
        if parameters and parameters[0].size > 1:
            json_str = bytes(parameters[0].tolist()).decode("utf-8")
            model_path = f"client_model_{self.client_id}.json"
            with open(model_path, "w", encoding="utf-8") as f:
                f.write(json_str)
            self.model.load_model(model_path)

        preds = self.model.predict(self.X_test)
        acc = accuracy_score(self.y_test, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(self.y_test, preds, average="binary", zero_division=0)

        # Confusion matrix counts for aggregation on server
        try:
            cm = confusion_matrix(self.y_test, preds, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
        except Exception:
            tn = fp = fn = tp = 0

        metrics = {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
        }
        # Flower expects a loss; we'll return 1-accuracy
        return float(1.0 - acc), len(self.X_test), metrics


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python client.py <client_id>")
        sys.exit(1)

    client_id = int(sys.argv[1])
    X_train, y_train, X_test, y_test = load_client_data(client_id)

    fl.client.start_client(
        server_address="172.16.174.136:8081",
        client=XGBClient(X_train, y_train, X_test, y_test, client_id).to_client()
    )
