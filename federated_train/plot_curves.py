#!/usr/bin/env python3
import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- Paths (zero-config) ----
RUN_DIR = "runs/cm_fi"
METRICS_CSV = os.path.join(RUN_DIR, "metrics.csv")
CM_CSV = os.path.join(RUN_DIR, "cm_latest.csv")
FI_CSV = os.path.join(RUN_DIR, "feature_importance_latest.csv")

SMOOTH_WINDOW = 1  # set >1 for moving average

# If you know the real feature names/order, set them here to label the FI plot:
FEATURE_COLS = [
    "Ammonia (ppm)", "Acetone (ppm)", "Ethanol (ppm)", "Hydrogen Sulfide (ppm)",
    "QoS", "message_length", "retained", "duplicate",
]

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def moving_avg(x, k):
    if k <= 1:
        return x
    return np.convolve(x, np.ones(k)/k, mode="valid")

def plot_curve(df, metric, out_dir, smooth=1):
    if metric not in df.columns:
        print(f"[skip] '{metric}' not found in metrics.csv")
        return
    rounds = df["round"].values
    vals = df[metric].astype(float).values

    if smooth > 1 and len(vals) >= smooth:
        vals = moving_avg(vals, smooth)
        rounds = rounds[smooth-1:]

    plt.figure()
    plt.plot(rounds, vals, marker="o")
    plt.title(f"{metric.capitalize()} over Rounds")
    plt.xlabel("Round")
    plt.ylabel(metric.capitalize())
    plt.grid(True)
    out_path = os.path.join(out_dir, f"{metric}_curve.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[ok] saved {out_path}")

def plot_confusion_from_counts(cm_csv, out_dir):
    if not os.path.exists(cm_csv):
        print(f"[skip] confusion matrix: {cm_csv} not found")
        return
    # Read tn, fp, fn, tp
    mp = {}
    with open(cm_csv, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            mp[row["cell"]] = int(row["count"])
    expected = {"tn","fp","fn","tp"}
    if not expected.issubset(mp.keys()):
        print(f"[skip] confusion matrix: missing keys in {cm_csv}")
        return

    cm = np.array([[mp["tn"], mp["fp"]],
                   [mp["fn"], mp["tp"]]])

    # Save numeric
    pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"])\
      .to_csv(os.path.join(out_dir, "confusion_matrix.csv"), index=True)

    # Plot heatmap
    plt.figure(figsize=(5.5, 4.5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Global Confusion Matrix (Aggregated)")
    plt.xticks([0,1], ["0","1"])
    plt.yticks([0,1], ["0","1"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    out_path = os.path.join(out_dir, "confusion_matrix.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[ok] saved {out_path}")

def plot_feature_importance(fi_csv, out_dir, feature_names=None):
    if not os.path.exists(fi_csv):
        print(f"[skip] feature importance: {fi_csv} not found")
        return
    df = pd.read_csv(fi_csv)
    if not {"feature_index","importance_gain"}.issubset(df.columns):
        print(f"[skip] feature importance: columns missing in {fi_csv}")
        return

    # If human names available and match length, use them
    weights = df["importance_gain"].astype(float).values
    if feature_names and len(feature_names) == len(weights):
        names = feature_names
    else:
        names = df["feature_index"].tolist()

    order = np.argsort(weights)
    plt.figure(figsize=(7.5, 4.8))
    plt.barh(np.array(names)[order], weights[order])
    plt.title("XGBoost Feature Importance (aggregated gain)")
    plt.xlabel("Importance")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "feature_importance_gain.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[ok] saved {out_path}")

def main():
    out_dir = ensure_dir(RUN_DIR)

    # Curves
    if not os.path.exists(METRICS_CSV):
        raise FileNotFoundError(f"{METRICS_CSV} not found")
    df = pd.read_csv(METRICS_CSV).sort_values("round").reset_index(drop=True)
    for metric in ["accuracy", "precision", "recall", "f1", "auc"]:
        plot_curve(df, metric, out_dir, smooth=SMOOTH_WINDOW)

    # Confusion matrix from aggregated counts
    plot_confusion_from_counts(CM_CSV, out_dir)

    # Feature importance aggregated across clients
    plot_feature_importance(FI_CSV, out_dir, feature_names=FEATURE_COLS)

if __name__ == "__main__":
    main()
