import numpy as np
from numpy.typing import NDArray


class Loss:
    def __init__(self):
        self.d_inputs = np.array([])

    def calculate(self, values: NDArray, expected: NDArray) -> float:
        sample = self.forward(values, expected)
        data_loss = np.mean(sample)

        return data_loss

    def forward(self, values: NDArray, expected: NDArray) -> NDArray:
        raise NotImplemented

    def backward(self, d_values: NDArray, expected: NDArray):
        raise NotImplemented


class Accuracy:
    @staticmethod
    def calculate(values: NDArray, expected: NDArray):
        predictions = np.argmax(values, axis=1)

        if len(expected.shape) == 2:
            targets = np.argmax(expected, axis=1)

            return np.mean(predictions == targets)

        return np.mean(predictions == expected)
