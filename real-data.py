import pandas as pd
import numpy as np

n_samples = 1000  # number of time steps
np.random.seed(42)

# Generate realistic ranges
O2 = np.random.normal(loc=98, scale=1.5, size=n_samples)              # Normal SpO2
NO = np.random.normal(loc=0.5, scale=0.2, size=n_samples)             # NO ~ micromolar
H2S = np.random.normal(loc=30, scale=5, size=n_samples)               # H2S ~ µM range
CO = np.random.normal(loc=3, scale=1.0, size=n_samples)               # CO low baseline

start_time = pd.Timestamp.now().floor('H')
timestamp = pd.date_range(start=start_time, periods=n_samples, freq='ms')

# Assemble DataFrame
real_data = pd.DataFrame({
    'timestamp': timestamp,
    'O2': O2,
    'NO': NO,
    'H2S': H2S,
    'CO': CO
})

real_data.to_csv("synthetic_wound_gas_flat.csv", index=False)

# Optional: Preview first few rows
print("✅ Saved as synthetic_wound_gas_flat.csv")

print(real_data.head())
