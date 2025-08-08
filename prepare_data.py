# prepare_data.py
from utils import load_partitioned_data

# This will preprocess the dataset and save files for each client
if __name__ == "__main__":
    load_partitioned_data(
        file_path="synthetic_wound_gas_dataset.csv",
        n_clients=2,
        test_size=0.3,
        output_dir="client_data"
    )
