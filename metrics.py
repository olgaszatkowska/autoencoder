import numpy as np
from numpy.typing import NDArray


class Loss:
    def __init__(self):
        self.d_inputs = np.array([])

    @staticmethod
    def calculate(self, values: NDArray, expected: NDArray) -> float:
        sample = self.forward(values, expected)
        data_loss = np.mean(sample)

        return data_loss

    def forward(self, values: NDArray, expected: NDArray) -> NDArray:
        raise NotImplemented

    def backward(self, d_values: NDArray, expected: NDArray):
        raise NotImplemented

 
class MeanSquaredError(Loss):
    # From Neural Networks from Scratch in Python
    # For regression
    def forward(self, values: NDArray, expected: NDArray) -> NDArray:
        return np.mean((expected - values) ** 2, axis=-1)

    def backward(self, d_values: NDArray, expected: NDArray):
        samples = len(d_values)
        outputs = len(d_values[0])

        self.d_inputs = -2 * (expected - d_values) / outputs
        self.d_inputs = self.d_inputs / samples
