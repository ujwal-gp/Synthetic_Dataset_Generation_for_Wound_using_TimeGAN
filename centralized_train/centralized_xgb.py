# centralized_xgb.py
import os, glob, time, json, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from typing import Tuple, List, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, label_binarize
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    average_precision_score,
    precision_recall_curve,
)
import xgboost as xgb

# -------------------------- IO / Utils --------------------------
def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def now_run_dir(base="runs") -> str:
    return ensure_dir(os.path.join(base, "central_xgb"))

def save_json(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def save_csv_row(path: str, row: dict, order: List[str]):
    exists = os.path.exists(path)
    import csv
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=order)
        if not exists:
            w.writeheader()
        w.writerow(row)

# -------------------------- Data discovery --------------------------
DEFAULT_LABEL_COL = "label"

def resolve_data_path(
    cli_path: Optional[str],
    env_var: str = "CENTRALIZED_XGB_DATA",
    auto_dirs: Optional[List[str]] = None,
) -> str:
    """Pick a CSV path with this priority:
    1) --data flag
    2) env CENTRALIZED_XGB_DATA
    3) ./data.csv
    4) ./data/data.csv
    5) first CSV in ./data/ containing a 'label' column
    """
    if cli_path and os.path.isfile(cli_path):
        return cli_path

    env_path = os.getenv(env_var)
    if env_path and os.path.isfile(env_path):
        return env_path

    candidates = [
        os.path.abspath("data.csv"),
        os.path.abspath(os.path.join("data", "data.csv")),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c

    search_dirs = auto_dirs or ["data"]
    for d in search_dirs:
        for p in sorted(glob.glob(os.path.join(d, "*.csv"))):
            try:
                # Peek only the header to check for label column quickly
                head = pd.read_csv(p, nrows=1)
                if DEFAULT_LABEL_COL in head.columns:
                    return p
            except Exception:
                pass

    raise FileNotFoundError(
        "No data CSV found.\n"
        "Place your file at ./data.csv or ./data/data.csv (with a 'label' column),\n"
        "or set env CENTRALIZED_XGB_DATA=/path/to/file.csv, or pass --data."
    )

# -------------------------- Loading --------------------------
def load_dataset(
    path: str,
    label_col: str = DEFAULT_LABEL_COL,
    test_size: float = 0.2,
    random_state: int = 42,
    scale_features: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    df = pd.read_csv(path)
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in {path}")

    X = df.drop(columns=[label_col])
    y = df[label_col].values

    X_num = X.select_dtypes(include=[np.number]).fillna(0)
    feature_names = list(X_num.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X_num.values,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if len(np.unique(y)) > 1 else None,
    )

    if scale_features:
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test, feature_names

# -------------------------- Plots --------------------------
def plot_confusion_matrix(cm: np.ndarray, labels, title: str, out_png: str):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.xticks(ticks=np.arange(len(labels)), labels=labels, rotation=45, ha="right")
    plt.yticks(ticks=np.arange(len(labels)), labels=labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def plot_feature_importance(fi: np.ndarray, feature_names: Optional[List[str]], out_png: str, top_k: int = 20, title: str = "Feature Importance (gain)"):
    idx = np.argsort(fi)[::-1][: min(top_k, len(fi))]
    names = [f"f{i}" for i in idx] if not feature_names else [feature_names[i] for i in idx]
    vals = fi[idx]
    plt.figure(figsize=(8, max(4, len(idx) * 0.35)))
    plt.barh(range(len(idx)), vals)
    plt.gca().invert_yaxis()
    plt.yticks(range(len(idx)), names)
    plt.title(title)
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def plot_roc_binary(y_true: np.ndarray, y_score: np.ndarray, auc_val: float, out_png: str, title="ROC Curve (binary)"):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc_val:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def plot_roc_multiclass(y_true_bin: np.ndarray, proba: np.ndarray, auc_val: float, out_png: str, title="ROC Curve (multiclass)"):
    fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), proba.ravel())
    fpr_grid = np.linspace(0.0, 1.0, 1001)
    k = proba.shape[1]
    tpr_accum = np.zeros_like(fpr_grid)
    for c in range(k):
        fpr_c, tpr_c, _ = roc_curve(y_true_bin[:, c], proba[:, c])
        tpr_accum += np.interp(fpr_grid, fpr_c, tpr_c, left=0, right=1)
    tpr_macro = tpr_accum / k
    plt.figure(figsize=(6, 5))
    plt.plot(fpr_micro, tpr_micro, label=f"micro-avg (AUC={auc_val:.3f})")
    plt.plot(fpr_grid, tpr_macro, label="macro-avg")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def plot_pr_curve_binary(y_true: np.ndarray, y_score: np.ndarray, ap_val: float, out_png: str, title="Precision–Recall (binary)"):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"AP = {ap_val:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

# -------------------------- Training / Eval --------------------------
def train_xgb(X_train: np.ndarray, y_train: np.ndarray, params: dict) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    return model

def evaluate_model(
    model: xgb.XGBClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    run_dir: str,
    feature_names: Optional[List[str]] = None,
    top_k_fi: int = 20,
) -> dict:
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average="weighted", zero_division=0)

    classes = sorted(np.unique(np.concatenate([y_test, preds])))
    cm = confusion_matrix(y_test, preds, labels=classes)

    booster = model.get_booster()
    gain_dict = booster.get_score(importance_type="gain")
    num_feats = X_train.shape[1]
    fi = np.zeros(num_feats, dtype=float)
    for k, v in gain_dict.items():
        try:
            idx = int(k[1:])
            if 0 <= idx < num_feats:
                fi[idx] = float(v)
        except Exception:
            pass

    auc_val = None
    ap_val = None
    roc_png = None
    pr_png = None
    try:
        proba = model.predict_proba(X_test)
        uniq = np.unique(y_test)
        if len(uniq) >= 2:
            if proba.shape[1] == 2:
                y_score = proba[:, 1]
                auc_val = float(roc_auc_score(y_test, y_score))
                ap_val = float(average_precision_score(y_test, y_score))
                roc_png = os.path.join(run_dir, "roc_curve.png")
                plot_roc_binary(y_test, y_score, auc_val, roc_png)
                pr_png = os.path.join(run_dir, "pr_curve.png")
                plot_pr_curve_binary(y_test, y_score, ap_val, pr_png)
            else:
                all_labels = np.unique(np.concatenate([y_train, y_test]))
                y_true_bin = label_binarize(y_test, classes=all_labels)
                auc_val = float(roc_auc_score(y_true_bin, proba, average="weighted", multi_class="ovr"))
                roc_png = os.path.join(run_dir, "roc_curve_multiclass.png")
                plot_roc_multiclass(y_true_bin, proba, auc_val, roc_png)
    except Exception:
        pass

    cm_png = os.path.join(run_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, classes, "Confusion Matrix", cm_png)
    save_json(os.path.join(run_dir, "confusion_matrix.json"), {"labels": list(map(int, classes)), "matrix": cm.tolist()})

    fi_png = os.path.join(run_dir, "feature_importance.png")
    plot_feature_importance(fi, feature_names, fi_png, top_k=top_k_fi)
    save_json(os.path.join(run_dir, "feature_importance.json"), fi.tolist())

    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "auc": None if auc_val is None else float(auc_val),
        "average_precision": None if ap_val is None else float(ap_val),
        "n_test": int(len(X_test)),
        "n_train": int(len(X_train)),
        "n_features": int(num_feats),
        "roc_png": roc_png,
        "pr_png": pr_png,
        "cm_png": cm_png,
        "fi_png": fi_png,
    }
    save_json(os.path.join(run_dir, "metrics.json"), metrics)
    save_csv_row(
        os.path.join(run_dir, "metrics.csv"),
        {k: metrics.get(k, "") for k in ["accuracy", "precision", "recall", "f1", "auc", "average_precision", "n_test", "n_train", "n_features"]},
        order=["accuracy", "precision", "recall", "f1", "auc", "average_precision", "n_test", "n_train", "n_features"],
    )

    model.save_model(os.path.join(run_dir, "model.json"))

    print(f"✅ Saved artifacts to: {run_dir}")
    print(
        f"Metrics — Acc: {metrics['accuracy']:.4f}, Prec: {metrics['precision']:.4f}, "
        f"Rec: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}, "
        f"AUC: {metrics['auc'] if metrics['auc'] is not None else 'NA'}"
    )
    return metrics

# -------------------------- Args & Main --------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Centralized XGBoost with full evaluation & plots")
    # All args are optional now
    p.add_argument("--data", type=str, default="../synthetic_wound_gas_dataset.csv", help="Path to CSV (defaults: env/file discovery)")
    p.add_argument("--label-col", type=str, default=DEFAULT_LABEL_COL, help="Label column name (default: 'label')")
    p.add_argument("--test-size", type=float, default=0.3, help="Test size (0..1), default 0.3")
    p.add_argument("--no-scale", action="store_true", help="Disable MinMax scaling")
    p.add_argument("--run-dir", type=str, default=None, help="Output directory (default: runs_central)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")

    # XGB hyperparams (sane defaults)
    p.add_argument("--n_estimators", type=int, default=100)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--learning-rate", type=float, default=0.1)
    p.add_argument("--subsample", type=float, default=1.0)
    p.add_argument("--colsample-bytree", type=float, default=1.0)
    p.add_argument("--reg-alpha", type=float, default=0.0)
    p.add_argument("--reg-lambda", type=float, default=1.0)
    return p.parse_args()

def main():
    args = parse_args()
    print(f'run_dir {args.run_dir}')

    # Resolve data path automatically if not provided
    data_path = resolve_data_path(args.data)
    print(f"📄 Using data: {data_path}")

    run_dir = ensure_dir(args.run_dir) if args.run_dir else now_run_dir()
    print(f'run_dir after {run_dir}')
    print(f"📂 Output directory: {run_dir}")

    X_train, y_train, X_test, y_test, feature_names = load_dataset(
        path=data_path,
        label_col=args.label_col,
        test_size=args.test_size,
        random_state=args.seed,
        scale_features=not args.no_scale,
    )

    xgb_params = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "reg_alpha": args.reg_alpha,
        "reg_lambda": args.reg_lambda,
        "objective": "binary:logistic" if len(np.unique(y_train)) == 2 else "multi:softprob",
        "eval_metric": "logloss",
        "verbosity": 1,
        "n_jobs": -1,
    }
    if xgb_params["objective"] == "multi:softprob":
        xgb_params["num_class"] = int(len(np.unique(np.concatenate([y_train, y_test]))))

    model = train_xgb(X_train, y_train, xgb_params)
    evaluate_model(model, X_train, y_train, X_test, y_test, run_dir, feature_names, top_k_fi=20)

if __name__ == "__main__":
    main()
