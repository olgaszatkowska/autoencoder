import numpy as np
from numpy.typing import NDArray


class ActivationFunction:
    def __init__(self):
        self.input = np.array([])
        self.output = np.array([])
        self.d_inputs = np.array([])

    def __str__(self):
        return self.__class__.__name__


class ReLU(ActivationFunction):
    def forward(self, inputs: NDArray) -> NDArray:
        self.input = inputs
        self.output = np.maximum(0, inputs)

        return self.output

    def backward(self, d_values: NDArray) -> None:
        self.d_inputs = d_values.copy()
        self.d_inputs[self.input <= 0] = 0


class Linear(ActivationFunction):
    def forward(self, values: NDArray) -> NDArray:
        self.output = values
        return self.output

    def backward(self, d_values: NDArray) -> NDArray:
        self.d_inputs = d_values.copy()
