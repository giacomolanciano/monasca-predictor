import logging
import pathlib

import numpy as np
import tensorflow as tf
import joblib
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression

log = logging.getLogger(__name__)


class Model:
    def __init__(self):
        self._model = None
        self._scaler = None
        self._model_dump_path = None
        self._scaler_dump_path = None

    def load(self, model_dump, scaler_dump=None):
        self._model_dump_path = pathlib.Path(model_dump)

        if self._model_dump_path.suffix == ".h5":
            self._model = tf.keras.models.load_model(model_dump)
        if self._model_dump_path.suffix == ".pt":
            # TODO: drop assumptions on underlying model.
            self._model = Model.RNN(2, 30, 1)
            self._model.load_state_dict(torch.load(model_dump))
            self._model.eval()
        elif self._model_dump_path.suffix == ".joblib":
            self._model = joblib.load(model_dump)
        else:
            raise ValueError(
                f"Unsupported model dump format '{self._model_dump_path.suffix}'."
            )

        log.debug("Loaded model of type '%s'.", type(self._model))

        if scaler_dump:
            self._scaler_dump_path = pathlib.Path(scaler_dump)

            if self._scaler_dump_path.suffix == ".joblib":
                self._scaler = joblib.load(self._scaler_dump_path)
            else:
                raise ValueError(
                    f"Unsupported scaler dump format '{self._scaler_dump_path.suffix}'."
                )

        log.debug("Loaded scaler of type '%s'.", type(self._scaler))

    def predict(self, x):
        if not isinstance(x, np.ndarray):
            log.debug("Input is of type '%s', converting to 'numpy.ndarray'.", type(x))
            x = np.array(x)

        x_length = len(x)

        # scaler expects 2D inputs
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        x_scaled = self._scaler.transform(x)

        log.debug("x_scaled (shape=%s):\n%s", str(x_scaled.shape), str(x_scaled))

        if self._model_dump_path.suffix == ".pt":
            hidden = self._model.init_hidden()
            for i in range(x_length):
                hidden, y_scaled = self._model(
                    torch.tensor(x_scaled[i], dtype=torch.float32), hidden
                )
            y_scaled = y_scaled.detach().numpy()
        else:
            # TODO: drop assumptions on underlying model input shape.
            # Currently assuming the underlying model is an LSTM expecting 3D
            # inputs.
            x_scaled = x_scaled.reshape(
                -1,
                *x_scaled.shape,
            )

            y_scaled = self._model.predict(x_scaled)

        log.debug("y_scaled (shape=%s): %s", str(y_scaled.shape), str(y_scaled))
        log.debug("scaler input shape: %s", str(self._scaler.scale_.shape))

        # NOTE: assuming the expected output of the model is a single value,
        # that is the last feature to be provided to the scaler.
        scaler_input_length = len(self._scaler.scale_)
        if y_scaled.shape[1] < scaler_input_length:
            y_scaled_temp = np.zeros((1, scaler_input_length))
            y_scaled_temp[0, -1] = y_scaled[0, -1]

            y = self._scaler.inverse_transform(y_scaled_temp)
            return float(y.flatten()[-1])

        return float(self._scaler.inverse_transform(y_scaled).flatten())

    class RNN(nn.Module):
        def __init__(self, in_size, hidden_size, out_size):
            super(Model.RNN, self).__init__()
            self.input_size = in_size
            self.hidden_size = hidden_size
            self.output_size = out_size
            self.i2h = nn.Linear(self.input_size + self.hidden_size, self.hidden_size)
            self.h2h = nn.LeakyReLU()
            self.i2o = nn.Linear(self.input_size + self.hidden_size, self.output_size)

        def forward(self, x, hidden):
            x = torch.unsqueeze(x, dim=1)
            # print(x.size())
            # print(hidden.size())
            concat = torch.cat((hidden, x), 0)
            # print(concat.size())
            # print(self.i2h)
            hidden = self.i2h(concat.T).T
            hidden = self.h2h(hidden)
            output = self.i2o(concat.T)
            return hidden, output

        def init_hidden(self):
            return torch.zeros(self.hidden_size, 1)


class LinearModel(Model):
    def __init__(self, prediction_offset):
        super().__init__()
        self._prediction_offset = prediction_offset

    def predict(self, x):
        if not isinstance(x, np.ndarray):
            log.debug("Input is of type '%s', converting to 'numpy.ndarray'.", type(x))
            x = np.array(x)

        # fit a linear model on input samples to get the trend
        time_steps = np.arange(0, len(x))
        self._model = LinearRegression().fit(time_steps.reshape(-1, 1), x)

        y = self._model.predict(np.array([[time_steps[-1] + self._prediction_offset]]))

        log.debug("y (shape=%s):\n%s", str(y.shape), str(y))

        # NOTE: assuming the expected output of the model is a single value
        return float(y.flatten())
