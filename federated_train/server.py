# server.py

import flwr as fl
import numpy as np
import os, json, csv
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

# headless-safe plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Settings
N_ROUNDS = 10
N_CLIENTS = 2
GLOBAL_MODEL_PATH = "global_model.json"
RUN_DIR = os.path.join("runs", "cm_fi")
os.makedirs(RUN_DIR, exist_ok=True)

FI_LATEST_CSV = os.path.join(RUN_DIR, "feature_importance_latest.csv")
CM_LATEST_CSV = os.path.join(RUN_DIR, "cm_latest.csv")
METRICS_CSV   = os.path.join(RUN_DIR, "metrics.csv")
ROC_POINTS_CSV = os.path.join(RUN_DIR, "roc_points.csv")
ROC_PNG        = os.path.join(RUN_DIR, "roc_curve.png")

def append_row_csv(path, row_dict, header_order=None):
    need_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header_order or list(row_dict.keys()))
        if need_header:
            writer.writeheader()
        writer.writerow(row_dict)

class Strategy(fl.server.strategy.FedAvg):
    def initialize_parameters(self, client_manager):
        # Cold start: let clients provide the initial model so shapes match
        return None

    def aggregate_fit(self, server_round, results, failures):
        """Pick best client update and aggregate feature importance."""
        print(f"🔁 Aggregating Round {server_round} from {len(results)} clients...")
        if not results:
            print("⚠️ No results to aggregate.")
            return None, {}

        # Choose best update by f1 (fallback accuracy)
        best_idx, best_key, best_val = None, None, -1.0
        fi_sum = None
        n_weight = 0

        for i, (_, fit_res) in enumerate(results):
            m = fit_res.metrics or {}
            key = "f1" if "f1" in m else ("accuracy" if "accuracy" in m else None)
            if key is not None:
                val = float(m[key])
                if val > best_val:
                    best_val, best_key, best_idx = val, key, i

            # Aggregate feature importance vectors (weighted by num_examples)
            fi_json = m.get("feature_importance_json")
            if fi_json is not None:
                try:
                    vec = np.array(json.loads(fi_json), dtype=float)
                    w = fit_res.num_examples or 0
                    if fi_sum is None:
                        fi_sum = np.zeros_like(vec, dtype=float)
                    fi_sum += w * vec
                    n_weight += w
                except Exception:
                    pass

        if best_idx is None:
            chosen_params = results[0][1].parameters
            print("⚠️ No metrics returned by clients; defaulting to first update.")
        else:
            chosen_params = results[best_idx][1].parameters
            print(f"✅ Chose client {best_idx} by best {best_key}={best_val:.4f}")

        # Save global model JSON for reproducibility (optional for plotting)
        arrays = parameters_to_ndarrays(chosen_params)
        buf = arrays[0]
        model_json = buf.tobytes().decode("utf-8")
        with open(GLOBAL_MODEL_PATH, "w", encoding="utf-8") as f:
            f.write(model_json)

        # Write latest aggregated feature importance (CSV)
        if fi_sum is not None and n_weight > 0:
            fi_avg = (fi_sum / max(1, n_weight)).tolist()
            with open(FI_LATEST_CSV, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["feature_index", "importance_gain"])
                for i, v in enumerate(fi_avg):
                    w.writerow([f"f{i}", float(v)])
            print(f"📝 Saved aggregated feature importance → {FI_LATEST_CSV}")

        return chosen_params, {}

    def aggregate_evaluate(self, server_round, results, failures):
        """Weighted average of metrics + sum confusion-matrix; build ROC from client histograms."""
        if not results:
            return None, {}

        total = 0
        sum_acc = sum_prec = sum_rec = sum_f1 = sum_auc = 0.0

        # CM counts (0.5 threshold)
        tp = fp = fn = tn = 0

        # ROC histogram accumulators (must be aligned across clients)
        thr_ref = None
        tp_vec = fp_vec = fn_vec = tn_vec = None

        for _, eval_res in results:
            n = eval_res.num_examples or 0
            m = eval_res.metrics or {}

            total += n
            sum_acc  += n * float(m.get("accuracy", 0.0))
            sum_prec += n * float(m.get("precision", 0.0))
            sum_rec  += n * float(m.get("recall", 0.0))
            sum_f1   += n * float(m.get("f1", 0.0))
            sum_auc  += n * float(m.get("auc", 0.0))

            tp += int(m.get("tp", 0))
            fp += int(m.get("fp", 0))
            fn += int(m.get("fn", 0))
            tn += int(m.get("tn", 0))

            # Aggregate thresholded counts for ROC
            rj = m.get("roc_counts_json")
            if rj:
                try:
                    data = json.loads(rj)
                    thr = np.array(data["thr"], dtype=float)
                    tp_c = np.array(data["tp"], dtype=float)
                    fp_c = np.array(data["fp"], dtype=float)
                    fn_c = np.array(data["fn"], dtype=float)
                    tn_c = np.array(data["tn"], dtype=float)
                    if thr_ref is None:
                        thr_ref = thr
                        tp_vec, fp_vec, fn_vec, tn_vec = tp_c, fp_c, fn_c, tn_c
                    else:
                        # ensure same thresholds; if not, skip this client's ROC
                        if np.allclose(thr_ref, thr):
                            tp_vec += tp_c
                            fp_vec += fp_c
                            fn_vec += fn_c
                            tn_vec += tn_c
                except Exception:
                    pass

        avg_acc  = sum_acc  / max(1, total)
        avg_prec = sum_prec / max(1, total)
        avg_rec  = sum_rec  / max(1, total)
        avg_f1   = sum_f1   / max(1, total)
        avg_auc  = sum_auc  / max(1, total)

        # Log metrics (now includes AUC)
        append_row_csv(
            METRICS_CSV,
            {"round": server_round, "accuracy": avg_acc, "precision": avg_prec, "recall": avg_rec, "f1": avg_f1, "auc": avg_auc},
            header_order=["round", "accuracy", "precision", "recall", "f1", "auc"],
        )
        print(f"📈 Round {server_round} — Acc={avg_acc:.4f}, Prec={avg_prec:.4f}, Rec={avg_rec:.4f}, F1={avg_f1:.4f}, AUC={avg_auc:.4f}")

        # Save latest confusion-matrix counts as CSV (0.5 threshold)
        with open(CM_LATEST_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["cell", "count"])
            w.writerow(["tn", tn])
            w.writerow(["fp", fp])
            w.writerow(["fn", fn])
            w.writerow(["tp", tp])
        print(f"📝 Saved aggregated confusion matrix counts → {CM_LATEST_CSV}")

        # Build and save ROC curve if we have aggregated histograms
        if thr_ref is not None:
            # FPR = FP / (FP + TN), TPR = TP / (TP + FN)
            with np.errstate(divide="ignore", invalid="ignore"):
                fpr = fp_vec / np.maximum(fp_vec + tn_vec, 1e-9)
                tpr = tp_vec / np.maximum(tp_vec + fn_vec, 1e-9)

            # sort by FPR ascending
            order = np.argsort(fpr)
            fpr = fpr[order]; tpr = tpr[order]; thr = thr_ref[order]

            # AUC from aggregated curve (geometric, not the same as avg AUC above)
            auc_geom = float(np.trapz(tpr, fpr))

            # Save points CSV
            with open(ROC_POINTS_CSV, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["threshold", "fpr", "tpr"])
                for a, b, c in zip(thr.tolist(), fpr.tolist(), tpr.tolist()):
                    w.writerow([a, b, c])

            # Plot ROC
            plt.figure()
            plt.plot(fpr, tpr, label=f"AUC≈{auc_geom:.3f}")
            plt.plot([0, 1], [0, 1], "k--")
            plt.xlim([0.0, 0.05])  # zoom in to first 5% FPR
            plt.ylim([0.95, 1.0])  # zoom in to high TPR zone
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("Global ROC (Aggregated from Clients)")
            plt.legend(loc="lower right")
            plt.grid(True)
            plt.savefig(ROC_PNG, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"📝 Saved ROC curve → {ROC_PNG} and points → {ROC_POINTS_CSV}")

        return None, {"accuracy": avg_acc, "precision": avg_prec, "recall": avg_rec, "f1": avg_f1, "auc": avg_auc}

def main():
    def fit_config(server_round: int):
        return {"rounds_to_add": 25}

    strategy = Strategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=N_CLIENTS,
        min_available_clients=N_CLIENTS,
        on_fit_config_fn=fit_config,
    )

    print("🚀 Starting Flower server...")
    fl.server.start_server(
        server_address="172.16.174.136:8081",
        config=fl.server.ServerConfig(num_rounds=N_ROUNDS),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
