from numpy.typing import NDArray
import tensorflow as ts
import numpy as np


def load_fashion_mnist_dataset(
    noisy: bool = False, noise_level: float = 0.1, y: bool = False
) -> list[NDArray]:

    (x_train, y_train), (x_test, y_test) = ts.keras.datasets.fashion_mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    if y:
        return x_train.reshape(-1, 784), y_train, x_test.reshape(-1, 784), y_test

    if noisy:
        noise = noise_level * np.random.randn(*x_train.shape)
        noisy_x_train = x_train + noise

        return noisy_x_train.reshape(-1, 784), x_train.reshape(-1, 784)

    return x_train.reshape(-1, 784), x_test.reshape(-1, 784)
