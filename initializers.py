import numpy as np
from numpy.typing import NDArray


class Initializer:
    def __init__(self, shape: NDArray, scale: float = 0.001) -> None:
        # (L-1, L)
        self._shape = shape
        self._scale = scale

    def initialize(self):
        pass

    def __str__(self) -> str:
        return self.__class__.__name__


class XavierInitializer(Initializer):
    # Xavier might be better than classic initializers (random and zeros).
    # It mitigates chances of issues with exploding gradients since the
    # weights are neither much bigger than 1, nor too much less than 1.
    # Source: https://towardsdatascience.com/weight-initialization-techniques-in-neural-networks-26c649eb3b78
    def initialize(self) -> NDArray:
        hidden_dim, _ = self._shape
        return np.random.randn(*self._shape) * np.sqrt(1 / hidden_dim)


class RandomInitializer(Initializer):
    def initialize(self) -> NDArray:
        return self._scale * np.random.rand(*self._shape)
