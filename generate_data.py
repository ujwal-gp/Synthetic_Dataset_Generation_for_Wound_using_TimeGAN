import os

from ydata.metadata import Metadata
from ydata.dataset.filetype import FileType
from ydata.connectors import LocalConnector
from ydata.utils.data_types import VariableType
from ydata.synthesizers.timeseries.model import TimeSeriesSynthesizer

os.environ['YDATA_LICENSE_KEY'] = '7ed1590a-07cf-4101-a598-ff57bdcd5513'

# Assemble DataFrame
connector = LocalConnector()
real_data = connector.read_file("synthetic_wound_gas_flat.csv", file_type=FileType.CSV, has_header=True)
print(real_data.columns)
real_data.astype("timestamp", VariableType.DATE)
dataset = real_data

print("\033[1m Dataset schema \033[0m")
print(dataset.schema)

if __name__ == "__main__":

    TRAIN = True
    SYNTHESIZE = True

    print(real_data.head())
    m = Metadata(real_data, dataset_attrs={"sortbykey": "timestamp"})
    # Getting the all metadata summary
    print("\n\033[1mMetadata summary\033[0m")
    print(m.summary)

    # Print the metadata
    print(m)

    if TRAIN is True:
        out_path = "./test_trained_model.pkl"
        synth = TimeSeriesSynthesizer()
        # Training configuration
        synth.fit(real_data, metadata=m)
        synth.save(out_path)

    if SYNTHESIZE is True:
        synth = TimeSeriesSynthesizer.load(out_path)
        n_entities = 10

        sample = synth.sample(n_entities=n_entities)
        print(f"Generated {len(sample)} samples.")

        sample.to_pandas().to_csv(r"test_synth_samples.csv")
