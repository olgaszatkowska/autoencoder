import pickle

from optimizer import Optimizer
from autoencoder import Autoencoder


def save_model(
    autoencoder: Autoencoder,
    autonecoder_name: str,
):
    with open(f"saved_models/{autonecoder_name}", "wb") as pickle_file:
        pickle.dump(autoencoder, pickle_file)


def open_model(
    autonecoder_name: str,
):
    with open(f"saved_models/{autonecoder_name}", "rb") as pickle_file:
        return pickle.load(pickle_file)
