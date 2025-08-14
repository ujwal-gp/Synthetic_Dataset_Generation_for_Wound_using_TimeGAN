from os import path
import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from ydata_synthetic.synthesizers.timeseries import TimeSeriesSynthesizer
from ydata_synthetic.synthesizers import ModelParameters, TrainParameters

# ========= CONFIG =========
DATA_CSV = "./mqtt_data/simulated_mqtt_gas_sensor_dataset_with_anomalies.csv"

# Labeling approach for synthetic:
#   "class_conditional" -> rows from class-0/1 synthesizers get label 0/1 (recommended)
#   "rules"             -> labels computed from feature thresholds (see label_rule)
LABEL_MODE = "class_conditional"  # "class_conditional" | "rules"

def label_rule(df: pd.DataFrame) -> pd.Series:
    # Tune thresholds to your real data distribution if you pick LABEL_MODE="rules"
    return (
        (df["Ammonia (ppm)"] > 25) |
        (df["Hydrogen Sulfide (ppm)"] > 10) |
        ((df["Acetone (ppm)"] > 15) & (df["Ethanol (ppm)"] > 20))
    ).astype(int)

RANDOM_STATE = 42
SYNTH_PER_CLASS = 1000

# Per-class checkpoints
MODEL_FILE_0 = "timegan_label0.pkl"
MODEL_FILE_1 = "timegan_label1.pkl"

# ---- Features to model (INCLUDE MQTT features) ----
FEATURE_COLS = [
    "Ammonia (ppm)", "Acetone (ppm)", "Ethanol (ppm)", "Hydrogen Sulfide (ppm)",
    "QoS", "message_length", "retained", "duplicate",
]

# ydata/TimeGAN hyperparams for tiny data
gan_args = ModelParameters(
    batch_size=16,
    lr=1e-3,
    noise_dim=16,
    layers_dim=64,
    latent_dim=12,
    gamma=1,
)
train_args = TrainParameters(
    epochs=300,
    sequence_length=8,
    number_sequences=8,
)

OUT_CSV = "synthetic_wound_gas_dataset.csv"
RUN_EVAL = True                       # Quick XGBoost sanity check

# ========= HELPERS =========
def to_int_safe(series, min_val=None, max_val=None, fill=0):
    s = pd.to_numeric(series, errors="coerce")
    if min_val is not None or max_val is not None:
        s = s.clip(lower=min_val if min_val is not None else s.min(),
                   upper=max_val if max_val is not None else s.max())
    s = s.round()
    if fill is not None:
        s = s.fillna(fill)
    return s.astype(int)

def to_bool_safe(series):
    if series.dtype == bool:
        return series
    s_num = pd.to_numeric(series, errors="coerce")
    if s_num.notna().any():
        return (s_num.fillna(0) >= 0.5)
    s_str = series.astype(str).str.strip().str.lower()
    return s_str.isin(["true", "1", "yes", "y", "t"])

def train_or_load(model_path, scaled_df, cols):
    if path.exists(model_path):
        print(f"🔄 Loading synthesizer: {model_path}")
        return TimeSeriesSynthesizer.load(model_path)
    if len(scaled_df) == 0:
        raise ValueError(f"No data to train synthesizer: {model_path}")
    print(f"🧪 Training synthesizer: {model_path} on {len(scaled_df)} rows...")
    synth = TimeSeriesSynthesizer(modelname="timegan", model_parameters=gan_args)
    synth.fit(pd.DataFrame(scaled_df, columns=cols), train_args, num_cols=cols)
    synth.save(model_path)
    return synth

def sample_and_inverse(synth, scaler, n_samples, cols):
    out = synth.sample(n_samples=n_samples)     # list of DataFrames
    flat = pd.concat(out, ignore_index=True)    # flatten
    flat[cols] = scaler.inverse_transform(flat[cols])
    return flat

# ========= LOAD DATA & PREPROCESS =========
df = pd.read_csv(DATA_CSV)

# Ensure required cols present
missing = [c for c in FEATURE_COLS + ["label"] if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns in CSV: {missing}")

# Keep metadata for optional reattach later
metadata_cols = [c for c in df.columns if c not in (FEATURE_COLS + ["label"])]

# Normalize types for MQTT fields BEFORE scaling
df["QoS"] = to_int_safe(df["QoS"], min_val=0, max_val=2, fill=0)
df["message_length"] = to_int_safe(df["message_length"], min_val=0, fill=0)
df["retained"] = to_bool_safe(df["retained"]).astype(int)   # 0/1
df["duplicate"] = to_bool_safe(df["duplicate"]).astype(int) # 0/1
df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).clip(0,1).astype(int)

# Work subset for modeling
df_model = df[FEATURE_COLS + ["label"]].copy()
print(f"✅ Using feature cols: {FEATURE_COLS}")
print("📊 Original class balance:", df_model["label"].value_counts().to_dict())

# Split by class
df0 = df_model[df_model["label"] == 0][FEATURE_COLS].copy()
df1 = df_model[df_model["label"] == 1][FEATURE_COLS].copy()

if len(df0) < 5 or len(df1) < 5:
    print("⚠️ Very few samples in one class. Consider reducing SYNTH_PER_CLASS or sequence_length, or adding more real data.")

# Per-class scalers on ALL features
scaler0 = MinMaxScaler().fit(df0) if len(df0) else None
scaler1 = MinMaxScaler().fit(df1) if len(df1) else None

df0_scaled = pd.DataFrame(scaler0.transform(df0), columns=FEATURE_COLS) if scaler0 is not None else pd.DataFrame(columns=FEATURE_COLS)
df1_scaled = pd.DataFrame(scaler1.transform(df1), columns=FEATURE_COLS) if scaler1 is not None else pd.DataFrame(columns=FEATURE_COLS)

# ========= TRAIN/LOAD SYNTHESIZERS =========
synth0 = train_or_load(MODEL_FILE_0, df0_scaled, FEATURE_COLS)
synth1 = train_or_load(MODEL_FILE_1, df1_scaled, FEATURE_COLS)

# ========= SAMPLE & INVERSE =========
print(f"🎲 Sampling {SYNTH_PER_CLASS} rows per class...")
s0 = sample_and_inverse(synth0, scaler0, SYNTH_PER_CLASS, FEATURE_COLS) if scaler0 is not None else pd.DataFrame(columns=FEATURE_COLS)
s1 = sample_and_inverse(synth1, scaler1, SYNTH_PER_CLASS, FEATURE_COLS) if scaler1 is not None else pd.DataFrame(columns=FEATURE_COLS)

# ========= LABEL ASSIGNMENT =========
if LABEL_MODE == "class_conditional":
    s0["label"] = 0
    s1["label"] = 1
    synth_df = pd.concat([s0, s1], ignore_index=True)
elif LABEL_MODE == "rules":
    synth_df = pd.concat([s0, s1], ignore_index=True)
    synth_df["label"] = label_rule(synth_df)
else:
    raise ValueError("LABEL_MODE must be 'class_conditional' or 'rules'.")

# ========= POST-PROCESS DISCRETE FIELDS =========
synth_df["QoS"] = to_int_safe(synth_df["QoS"], min_val=0, max_val=2, fill=0)
synth_df["message_length"] = to_int_safe(synth_df["message_length"], min_val=0, fill=0)
synth_df["retained"] = to_bool_safe(synth_df["retained"]).astype(bool)
synth_df["duplicate"] = to_bool_safe(synth_df["duplicate"]).astype(bool)

# Shuffle
synth_df = synth_df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
print("📊 Synthetic class balance:", synth_df["label"].value_counts().to_dict())

# ========= OPTIONAL METADATA REATTACH =========
out_df = pd.concat(
    [df[metadata_cols].reset_index(drop=True),
        synth_df[FEATURE_COLS + ["label"]].reset_index(drop=True)],
    axis=1
)
out_df = out_df[df.columns]  # match original order

# ========= SAVE =========
out_df.to_csv(OUT_CSV, index=False)
print(f"✅ Saved synthetic dataset: {OUT_CSV} shape: {out_df.shape}")

# ========= QUICK SANITY EVAL =========
if True:
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        from xgboost import XGBClassifier

        eval_df = out_df.copy()
        if eval_df["retained"].dtype == bool:
            eval_df["retained"] = eval_df["retained"].astype(int)
        if eval_df["duplicate"].dtype == bool:
            eval_df["duplicate"] = eval_df["duplicate"].astype(int)

        X = eval_df[FEATURE_COLS].values
        y = eval_df["label"].values

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
        )

        clf = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.15,
            subsample=0.9, colsample_bytree=0.9, tree_method="hist",
            n_jobs=-1, scale_pos_weight=((y_tr == 0).sum() / max(1, (y_tr == 1).sum()))
        )
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)

        acc = accuracy_score(y_te, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_te, y_pred, average="binary", zero_division=0)
        print(f"🧪 Quick check — Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")
    except Exception as e:
        print(f"Eval skipped (install xgboost to run): {e}")
