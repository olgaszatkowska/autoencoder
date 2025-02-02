from numpy.typing import NDArray

from neural_network import Autoencoder


def deflatten(ndarray: NDArray):
    return ndarray.reshape(28, 28)


def reconstruct_image(image: NDArray, autoencoder: Autoencoder):
    reconstructed = autoencoder.forward(image)

    return deflatten(reconstructed)
