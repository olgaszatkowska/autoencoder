import numpy as np
from numpy.typing import NDArray

from neural_network import CustomizedNeuralNetwork
from activation_functions import ReLU, Linear
from initializers import XavierInitializer, RandomInitializer


class Autoencoder(CustomizedNeuralNetwork):
    def __init__(
        self,
        *,
        input_dim: int,
        code_dim: int,
        encoder_hidden_count: int,
        reduce_by: int,
    ) -> None:
        super().__init__(input_dim=input_dim)
        self.code_dim = code_dim
        self._add_layers(encoder_hidden_count, reduce_by)

    def _add_layers(self, encoder_hidden_count: int, reduce_by: int):
        self.add_layer(self.input_dim, ReLU(), XavierInitializer)

        hidden_dim = self.input_dim // reduce_by
        saved_dims = []

        for _ in range(encoder_hidden_count):
            hidden_dim = int(hidden_dim // reduce_by)
            saved_dims.append(hidden_dim)
            self.add_layer(hidden_dim, ReLU(), RandomInitializer)

        self.add_layer(self.code_dim, ReLU(), XavierInitializer)

        for dim in reversed(saved_dims):
            self.add_layer(dim, ReLU(), RandomInitializer)

        self.add_layer(self.input_dim, Linear(), XavierInitializer)

    def get_encoded(self, inputs: NDArray, slice: int):
        vector = inputs

        for layer, activation_fn in zip(
            self.layers[:slice], self.activation_fns[:slice]
        ):
            layer.forward(vector)
            activation_fn.forward(layer.output)

            vector = activation_fn.output

        return vector
