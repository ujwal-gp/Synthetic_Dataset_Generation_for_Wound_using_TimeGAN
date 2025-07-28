from os import path
from ydata_synthetic.synthesizers.timeseries import TimeSeriesSynthesizer
from sklearn.preprocessing import MinMaxScaler
from ydata_synthetic.synthesizers import ModelParameters, TrainParameters
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define model parameters
gan_args =  ModelParameters(
    batch_size=128,
    lr=5e-4,
    noise_dim=32,
    layers_dim=128,
    latent_dim=24,
    gamma=1)

train_args = TrainParameters(
    epochs=1000,
    sequence_length=24,
    number_sequences=4)

# Read the data
wound_gas_data = pd.read_csv("synthetic_wound_gas_data.csv")
cols = list(wound_gas_data.columns)

model_file = 'synthetic_wound_gas_data.pkl'

# Training the TimeGAN synthesizer
if path.exists(model_file):
    synth = TimeSeriesSynthesizer.load(model_file)
else:
    synth = TimeSeriesSynthesizer(modelname='timegan', model_parameters=gan_args)
    synth.fit(wound_gas_data, train_args, num_cols=cols)
    synth.save(model_file)

# Generating new synthetic samples
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(wound_gas_data[cols])
synth_data = synth.sample(n_samples=len(scaled_data))

# Plotting some generated samples. Both Synthetic and Original data are still standartized with values between [0,1]
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 10))
axes=axes.flatten()

def create_sequences(data, seq_len):
    return np.array([data[i:i + seq_len] for i in range(len(data) - seq_len)])

real_sequences = create_sequences(scaled_data, 24)

obs = np.random.randint(len(real_sequences))

for j, col in enumerate(cols):
    real = real_sequences[obs][:, j]
    synth_sample = synth_data[obs].iloc[:, j]
    df = pd.DataFrame({'Real': real,
                   'Synthetic': synth_sample})
    df.plot(ax=axes[j],
            title = col,
            secondary_y='Synthetic data', style=['-', '--'])
fig.tight_layout()
plt.show()