from numpy.typing import NDArray

from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity


def mse(values: NDArray, expected: NDArray) -> float:
    return mean_squared_error(expected.flatten(), values.flatten())


def ssim(values: NDArray, expected: NDArray) -> float:
    return structural_similarity(expected, values, data_range=1.0)
