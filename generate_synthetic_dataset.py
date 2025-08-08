from os import path
from ydata_synthetic.synthesizers.timeseries import TimeSeriesSynthesizer
from sklearn.preprocessing import MinMaxScaler
from ydata_synthetic.synthesizers import ModelParameters, TrainParameters
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- Config ----------------
LABEL_MODE = "sample"  # "copy" | "sample" | "rules"
RANDOM_STATE = 42
model_file = "synthetic_wound_gas_data.pkl"  # pkl works well with ydata_synthetic

# Load original dataset
wound_gas_data = pd.read_csv("./mqtt_data/simulated_mqtt_gas_sensor_dataset_with_anomalies.csv")

# Identify numeric feature columns to model (EXCLUDE 'label' and any strings)
numeric_cols = ["Ammonia (ppm)", "Acetone (ppm)", "Ethanol (ppm)", "Hydrogen Sulfide (ppm)"]
numeric_cols = [c for c in numeric_cols if c != "label" and c in wound_gas_data.columns]
print(f"numeric columns {numeric_cols}")

# Metadata = everything not modeled by TimeGAN
metadata_cols = [c for c in wound_gas_data.columns if c not in numeric_cols]

# Prepare numeric data
numeric_data = wound_gas_data[numeric_cols].copy()
cols = numeric_data.columns.tolist()

# Sanity checks
n_features = numeric_data.shape[1]
assert len(cols) == n_features, f"cols({len(cols)}) != num_df.shape[1]({n_features})"
print("Training with n_features =", n_features)

# Scale numeric data (so we can inverse later)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(numeric_data)

# Define model parameters
gan_args = ModelParameters(
    batch_size=128,
    lr=5e-4,
    noise_dim=32,
    layers_dim=128,
    latent_dim=24,
    gamma=1,
)

train_args = TrainParameters(
    epochs=50,
    sequence_length=24,
    number_sequences=4,
)

# Train or load model
if path.exists(model_file):
    synth = TimeSeriesSynthesizer.load(model_file)
else:
    synth = TimeSeriesSynthesizer(modelname="timegan", model_parameters=gan_args)
    # Train on scaled numeric data
    synth.fit(pd.DataFrame(scaled_data, columns=numeric_cols), train_args, num_cols=numeric_cols)
    synth.save(model_file)

# Generate synthetic samples (in scaled space)
synth_data = synth.sample(n_samples=len(scaled_data))

# Flatten and inverse scale to original range
flat_synth_data = pd.concat(synth_data, ignore_index=True)
flat_synth_data[numeric_cols] = scaler.inverse_transform(flat_synth_data[numeric_cols])

# Reattach metadata columns (row-wise copy to preserve structure)
synthetic_full_df = pd.concat(
    [
        wound_gas_data[metadata_cols].reset_index(drop=True),
        flat_synth_data[numeric_cols].reset_index(drop=True),
    ],
    axis=1,
)

# ---------- Discrete field cleanup (robust) ----------
def to_int_safe(series, min_val=None, max_val=None, fill=0):
    s = pd.to_numeric(series, errors="coerce")
    if min_val is not None and max_val is not None:
        s = s.clip(min_val, max_val)
    s = s.round()
    if fill is not None:
        s = s.fillna(fill)
    return s.astype(int)

def to_bool_safe(series):
    if series.dtype == bool:
        return series
    # Try numeric first
    s_num = pd.to_numeric(series, errors="coerce")
    if s_num.notna().any():
        return (s_num.fillna(0) >= 0.5)
    # Fallback: strings like "true"/"false"
    s_str = series.astype(str).str.strip().str.lower()
    return s_str.isin(["true","1","yes","y","t"])

# QoS to {0,1,2}, fill missing with 0
if "QoS" in synthetic_full_df.columns:
    synthetic_full_df["QoS"] = to_int_safe(synthetic_full_df["QoS"], min_val=0, max_val=2, fill=0)

if "message_length" in synthetic_full_df.columns:
    synthetic_full_df["message_length"] = (
        pd.to_numeric(synthetic_full_df["message_length"], errors="coerce")
          .fillna(0)          # replace NaN with 0
          .clip(lower=0)      # no negatives
          .round()            # nearest integer
          .astype(int)        # convert to int
    )

# Booleans
for b in ("retained", "duplicate"):
    if b in synthetic_full_df.columns:
        synthetic_full_df[b] = to_bool_safe(synthetic_full_df[b]).astype(bool)

# Labels: ALWAYS 0/1
if "label" in wound_gas_data.columns:
    if LABEL_MODE == "copy":
        synthetic_full_df["label"] = (
            pd.to_numeric(wound_gas_data["label"], errors="coerce")
            .fillna(0).clip(0, 1).round().astype(int)
        )
    elif LABEL_MODE == "sample":
        p = pd.to_numeric(wound_gas_data["label"], errors="coerce").fillna(0).clip(0,1).mean()
        rng = np.random.default_rng(RANDOM_STATE)
        synthetic_full_df["label"] = rng.binomial(1, p, size=len(synthetic_full_df)).astype(int)
    elif LABEL_MODE == "rules":
        # synthetic_full_df["label"] = (<your boolean condition>).astype(int)
        raise ValueError("Provide your labeling rules in the 'rules' branch.")
    else:
        synthetic_full_df["label"] = to_int_safe(synthetic_full_df["label"], min_val=0, max_val=1, fill=0)

# Ensure same column order as original
synthetic_full_df = synthetic_full_df[wound_gas_data.columns]

# Save final CSV
synthetic_full_df.to_csv("synthetic_wound_gas_dataset.csv", index=False)
print("✅ Synthetic dataset saved with exact original structure:", synthetic_full_df.shape)
