from typing import Optional
from numpy.typing import NDArray
import numpy as np
import copy

from neural_network import BaseNeuralNetwork
from metrics import Loss


class Optimizer:
    def __init__(
        self,
        network: BaseNeuralNetwork,
        loss: Loss,
        learning_rate: float,
        batch_size: int,
        epochs: int,
        reshape: bool = False,
        early_stopping: bool = False,
    ):
        self.network = network
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

        self.loss_fn = loss

        self.loss = []
        self.validation_loss = []

        self.reshape = reshape

        self.early_stopping = early_stopping

    def fit(
        self, x: NDArray, y: NDArray, x_valid: NDArray = None, y_valid: NDArray = None
    ):
        batch_count = x.shape[0] // self.batch_size
        validation_loss = None
        saved_model = None

        for epoch in range(self.epochs):
            if self.early_stopping:
                saved_model = copy.deepcopy(self.network)

            loss = self._fit_batch(x, y, batch_count)
            self._print_metrics(epoch, loss)

            if not self.early_stopping:
                continue

            predictions = self.network.forward(x_valid)
            validation_loss_new = self.loss_fn.calculate(predictions, y_valid)

            if self.early_stopping and validation_loss != None:
                self.validation_loss.append(validation_loss_new)

                if (validation_loss_new - validation_loss) > 0:
                    self.network = saved_model
                    print(f"Early stopping at {validation_loss_new}")
                    return

            validation_loss = validation_loss_new

    def _fit_batch(self, x: NDArray, y: NDArray, batch_count: int):
        loss = 0.0, 0.0
        batch_loss = []

        for batch_no in range(batch_count):
            x_window, y_window = self._get_window(batch_no, x, y)
            predictions = self.network.forward(x_window)

            loss = self.loss_fn.calculate(predictions, y_window)
            batch_loss.append(loss)

            self._backward(predictions, y_window)

        avg_loss = np.average(batch_loss)

        self._append_metrics(avg_loss)

        return avg_loss

    def _get_window(self, batch_no: int, x: NDArray, y: NDArray):
        start_idx = batch_no * self.batch_size
        end_idx = start_idx + self.batch_size

        y_window = y[start_idx:end_idx]

        if self.reshape:
            return x[start_idx:end_idx], y_window.reshape(-1, 1)

        return x[start_idx:end_idx], y_window

    def _backward(self, predictions: NDArray, y_window: NDArray):
        self.loss_fn.backward(predictions, y_window)
        self.network.backwards(self.loss_fn.d_inputs)
        for layer in self.network.layers:
            layer.weights += -self.learning_rate * layer.d_weights
            layer.biases += -self.learning_rate * layer.d_bias

    def _append_metrics(self, loss: float):
        self.loss.append(loss)

    def _print_metrics(self, epoch: int, loss: float) -> None:
        print(f"Epoch {epoch}  --  loss {loss:.3f}")
