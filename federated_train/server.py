# server.py

import flwr as fl
import numpy as np
import os, json, csv, time
import matplotlib.pyplot as plt
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

# Settings
N_ROUNDS = 10
N_CLIENTS = 2
GLOBAL_MODEL_PATH = "global_model.json"
RUN_DIR = os.path.join("runs", "cm_fi")
os.makedirs(RUN_DIR, exist_ok=True)

# ---- 1. Initialize a Dummy XGBoost Model (once) ----
import xgboost as xgb

def initialize_global_model():
    model = xgb.XGBClassifier(n_estimators=200, max_depth=3)
    X_dummy = np.array([[0, 0], [1, 1]])
    y_dummy = np.array([0, 1])
    model.fit(X_dummy, y_dummy)

    model.get_booster().save_model(GLOBAL_MODEL_PATH)
    print("📦 Initial global model saved.")

def save_confusion_matrix(cm: np.ndarray, labels, out_path: str):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest')
    plt.title("Global Confusion Matrix")
    plt.xticks(ticks=np.arange(len(labels)), labels=labels, rotation=45, ha="right")
    plt.yticks(ticks=np.arange(len(labels)), labels=labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def save_feature_importance(fi: np.ndarray, out_path: str, feature_names=None, top_k=20):
    idx = np.argsort(fi)[::-1]
    idx = idx[:min(top_k, len(idx))]
    names = [f"f{i}" for i in idx] if feature_names is None else [feature_names[i] for i in idx]
    vals = fi[idx]

    plt.figure(figsize=(8, max(4, len(idx) * 0.3)))
    plt.barh(range(len(idx)), vals)
    plt.gca().invert_yaxis()
    plt.yticks(range(len(idx)), names)
    plt.title("Global Feature Importance (avg gain)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def append_row_csv(csv_path, row_dict, header_order):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header_order)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)

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
    def aggregate_evaluate(self, server_round, results, failures):
        # results: List[Tuple[ClientProxy, EvaluateRes]]
        if not results:
            return None, {}

        total_examples = 0
        sum_acc = sum_prec = sum_rec = sum_f1 = sum_auc = 0.0

        # For confusion matrix we need to align labels across clients
        global_labels = None
        global_cm = None

        # For feature importance (vector)
        fi_sum = None

        for _, eval_res in results:
            m = eval_res.metrics
            n = int(m.get("num_test_examples", eval_res.num_examples))
            total_examples += n

            # weighted sums
            sum_acc  += n * float(m.get("accuracy", 0.0))
            sum_prec += n * float(m.get("precision", 0.0))
            sum_rec  += n * float(m.get("recall", 0.0))
            sum_f1   += n * float(m.get("f1", 0.0))
            sum_auc  += n * float(m.get("auc", 0.0))

            # --- confusion matrix aggregation ---
            if "confusion_matrix_json" in m:
                cm_payload = json.loads(m["confusion_matrix_json"])
                labels = cm_payload["labels"]
                cm = np.array(cm_payload["matrix"], dtype=np.int64)

                if global_labels is None:
                    # Initialize
                    global_labels = labels
                    global_cm = cm.copy()
                else:
                    # Align if label orders differ
                    # Build mapping from client label index -> global index
                    label_to_idx = {lbl: i for i, lbl in enumerate(global_labels)}
                    # Expand global cm if new labels appear
                    for lbl in labels:
                        if lbl not in label_to_idx:
                            # append new label
                            global_labels.append(lbl)
                            # expand cm
                            new_size = len(global_labels)
                            new_mat = np.zeros((new_size, new_size), dtype=np.int64)
                            new_mat[:global_cm.shape[0], :global_cm.shape[1]] = global_cm
                            global_cm = new_mat
                            label_to_idx = {lbl2: i for i, lbl2 in enumerate(global_labels)}

                    # Now add this client's cm into the correct indices
                    for i_client, lbl_i in enumerate(labels):
                        for j_client, lbl_j in enumerate(labels):
                            gi = label_to_idx[lbl_i]
                            gj = label_to_idx[lbl_j]
                            global_cm[gi, gj] += cm[i_client, j_client]

            # --- feature importance aggregation ---
            if "feature_importance_json" in m:
                fi_vec = np.array(json.loads(m["feature_importance_json"]), dtype=float)
                if fi_sum is None:
                    fi_sum = np.zeros_like(fi_vec, dtype=float)
                # Simple average later (equal weight per client) or weight by n:
                fi_sum += fi_vec  # equal client weight
                # If you prefer weighting by examples, do: fi_sum += n * fi_vec

        # --- finalize aggregates ---
        k_clients = max(1, len(results))
        avg_acc  = sum_acc  / max(1, total_examples)
        avg_prec = sum_prec / max(1, total_examples)
        avg_rec  = sum_rec  / max(1, total_examples)
        avg_f1   = sum_f1   / max(1, total_examples)
        avg_auc  = sum_auc / max(1, total_examples)

        agg_metrics = {
            "accuracy": float(avg_acc),
            "precision": float(avg_prec),
            "recall": float(avg_rec),
            "f1": float(avg_f1),
            "auc": float(avg_auc),
        }

        metrics_csv = os.path.join(RUN_DIR, "metrics.csv")
        append_row_csv(
            metrics_csv,
            {"round": server_round, "accuracy": avg_acc, "precision": avg_prec, "recall": avg_rec, "f1": avg_f1},
            header_order=["round", "accuracy", "precision", "recall", "f1"],
        )

        if fi_sum is not None and k_clients > 0:
            fi_avg = (fi_sum / k_clients)
            agg_metrics["feature_importance_json"] = json.dumps(fi_avg.tolist())
            fi_png = os.path.join(RUN_DIR, f"fi_round_{server_round:03d}.png")
            save_feature_importance(fi_avg, fi_png, feature_names=None, top_k=20)

        if global_cm is not None and global_labels is not None:
            agg_metrics["global_confusion_matrix_json"] = json.dumps({
                "labels": list(map(int, global_labels)),
                "matrix": global_cm.tolist(),
            })
            cm_png = os.path.join(RUN_DIR, f"cm_round_{server_round:03d}.png")
            save_confusion_matrix(global_cm, global_labels, cm_png)

        # (loss, metrics) required by Flower; loss here can be 1-acc
        agg_loss = float(1.0 - avg_acc)
        return agg_loss, agg_metrics

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
