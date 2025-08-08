# client.py
import flwr as fl
import xgboost as xgb
import numpy as np
import os
import json
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import MinMaxScaler, label_binarize
import sys

# --- plotting (headless-safe) ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Load client-specific data
def load_client_data(client_id, data_dir="../client_data"):
    train_file = os.path.join(data_dir, f"client_{client_id}_train.csv")
    test_file = os.path.join(data_dir, f"client_{client_id}_test.csv")

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    # Split into features and labels
    X_train = train_df.drop("label", axis=1)
    y_train = train_df["label"]
    X_test = test_df.drop("label", axis=1)
    y_test = test_df["label"]

    # Filter numeric columns only and scale them
    X_train_numeric = X_train.select_dtypes(include=[np.number]).fillna(0)
    X_test_numeric = X_test.select_dtypes(include=[np.number]).fillna(0)

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
        self.model = xgb.XGBClassifier()

    def get_parameters(self, config):
        model_path = f"client_model_{self.client_id}.json"
        self.model.save_model(model_path)  # Always save in JSON format
        with open(model_path, "r", encoding="utf-8") as f:
            model_json_str = f.read()
        return [np.frombuffer(model_json_str.encode("utf-8"), dtype=np.uint8)]

    def fit(self, parameters, config):
        if parameters and parameters[0].size > 1:
            json_str = bytes(parameters[0].tolist()).decode("utf-8")
            model_path = f"client_model_{self.client_id}.json"
            with open(model_path, "w", encoding="utf-8") as f:
                f.write(json_str)
            self.model.load_model(model_path)

        self.model.fit(self.X_train, self.y_train)

        return self.get_parameters(config), len(self.X_train), {}

    def evaluate(self, parameters, config):
        if parameters and parameters[0].size > 1:
            json_str = bytes(parameters[0].tolist()).decode("utf-8")
            model_path = f"client_model_{self.client_id}.json"
            with open(model_path, "w", encoding="utf-8") as f:
                f.write(json_str)
            self.model.load_model(model_path)

        preds = self.model.predict(self.X_test)

        acc = accuracy_score(self.y_test, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(
            self.y_test, preds, average="weighted", zero_division=0
        )

        # --- confusion matrix (sum on server) ---
        classes = sorted(np.unique(np.concatenate([self.y_test, preds])))
        cm = confusion_matrix(self.y_test, preds, labels=classes)  # shape [C,C]

        # --- feature importance (gain) aligned to feature count ---
        # XGBoost names features as f0, f1, ... in the booster
        booster = self.model.get_booster()
        gain_dict = booster.get_score(importance_type="gain")  # {"f0": val, ...}
        num_feats = self.X_train.shape[1]
        fi = np.zeros(num_feats, dtype=float)
        for k, v in gain_dict.items():
            try:
                idx = int(k[1:])  # "f12" -> 12
                if idx < num_feats:
                    fi[idx] = v
            except Exception:
                pass

        # ---- AUC + ROC (binary or multiclass) ----
        auc_val = 0.0
        try:
            proba = self.model.predict_proba(self.X_test)
            unique_classes = np.unique(self.y_test)

            # Guard against single-class y_test (AUC undefined)
            if len(unique_classes) >= 2:
                if proba.shape[1] == 2:  # binary
                    y_score = proba[:, 1]
                    auc_val = float(roc_auc_score(self.y_test, y_score))
                    # ROC curve for binary
                    fpr, tpr, _ = roc_curve(self.y_test, y_score)
                    roc_curves = {"fpr": fpr, "tpr": tpr}
                    is_multiclass = False
                else:  # multiclass
                    all_labels = np.unique(np.concatenate([self.y_train, self.y_test]))
                    y_true_bin = label_binarize(self.y_test, classes=all_labels)
                    # weighted OVR AUC
                    auc_val = float(
                        roc_auc_score(y_true_bin, proba, average="weighted", multi_class="ovr")
                    )
                    # micro/macro ROC
                    # micro
                    fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), proba.ravel())
                    # macro: average interpolated ROC across classes
                    fpr_grid = np.linspace(0, 1, 1001)
                    tpr_accum = np.zeros_like(fpr_grid)
                    k = proba.shape[1]
                    for c in range(k):
                        fpr_c, tpr_c, _ = roc_curve(y_true_bin[:, c], proba[:, c])
                        tpr_accum += np.interp(fpr_grid, fpr_c, tpr_c, left=0, right=1)
                    tpr_macro = tpr_accum / k
                    roc_curves = {
                        "fpr_micro": fpr_micro,
                        "tpr_micro": tpr_micro,
                        "fpr_macro": fpr_grid,
                        "tpr_macro": tpr_macro,
                    }
                    is_multiclass = True
            else:
                roc_curves = None
        except Exception:
            # If predict_proba not available or anything weird: leave AUC=0 and skip ROC
            roc_curves = None

        print(f"📊 Client {self.client_id} Evaluation — Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, AUC: {auc_val:.4f}")

        # ---------------- Local saves: PNG + JSON ----------------
        run_dir = os.path.join(
            "runs_client", f"client_{self.client_id}"
        )
        os.makedirs(run_dir, exist_ok=True)

        # Confusion matrix PNG
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, interpolation="nearest")
        plt.title(f"Client {self.client_id} — Confusion Matrix")
        plt.xticks(ticks=np.arange(len(classes)), labels=classes, rotation=45, ha="right")
        plt.yticks(ticks=np.arange(len(classes)), labels=classes)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        cm_png = os.path.join(run_dir, "confusion_matrix.png")
        plt.savefig(cm_png, dpi=150)
        plt.close()

        # Confusion matrix JSON
        cm_json = {
            "labels": list(map(int, classes)),
            "matrix": cm.tolist(),
        }
        with open(os.path.join(run_dir, "confusion_matrix.json"), "w", encoding="utf-8") as f:
            json.dump(cm_json, f, ensure_ascii=False, indent=2)

        # Feature importance PNG (top-k)
        top_k = min(20, num_feats)
        order = np.argsort(fi)[::-1][:top_k]
        names = [f"f{i}" for i in order]
        vals = fi[order]

        plt.figure(figsize=(8, max(4, len(order) * 0.3)))
        plt.barh(range(len(order)), vals)
        plt.gca().invert_yaxis()
        plt.yticks(range(len(order)), names)
        plt.title(f"Client {self.client_id} — Feature Importance (gain)")
        plt.xlabel("Importance")
        plt.tight_layout()
        fi_png = os.path.join(run_dir, "feature_importance.png")
        plt.savefig(fi_png, dpi=150)
        plt.close()

        # Feature importance JSON (full vector)
        with open(os.path.join(run_dir, "feature_importance.json"), "w", encoding="utf-8") as f:
            json.dump(fi.tolist(), f, ensure_ascii=False)

        print(f"[Client {self.client_id}] Saved CM → {cm_png}")
        print(f"[Client {self.client_id}] Saved FI → {fi_png}")
        # ---------------------------------------------------------

        # ROC curve
        if roc_curves is not None:
            plt.figure(figsize=(6, 5))
            if not is_multiclass:
                plt.plot(roc_curves["fpr"], roc_curves["tpr"], label=f"ROC (AUC={auc_val:.3f})")
            else:
                plt.plot(
                    roc_curves["fpr_micro"],
                    roc_curves["tpr_micro"],
                    label=f"micro-avg ROC (AUC={auc_val:.3f})",
                )
                plt.plot(roc_curves["fpr_macro"], roc_curves["tpr_macro"], label="macro-avg ROC")
            plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"Client {self.client_id} — ROC Curve")
            plt.legend()
            plt.tight_layout()
            roc_png = os.path.join(run_dir, "roc_curve.png")
            plt.savefig(roc_png, dpi=150)
            plt.close()
        # ---------------------------------------------------------

        # IMPORTANT: metrics dict must be scalars/str/bytes. Encode arrays as JSON strings.
        metrics = {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "auc": float(auc_val),
            # include sizes so server can do weighted averaging
            "num_test_examples": int(len(self.X_test)),

            # JSON-encoded arrays
            "confusion_matrix_json": json.dumps({
                "labels": list(map(int, classes)),
                "matrix": cm.tolist(),
            }),
            "feature_importance_json": json.dumps(fi.tolist()),
        }

        # Flower expects (loss, num_examples, metrics). Use 1-acc as loss proxy.
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
