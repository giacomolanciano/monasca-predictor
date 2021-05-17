import logging
import pathlib

import numpy as np
import tensorflow as tf
import joblib
from sklearn.linear_model import LinearRegression

log = logging.getLogger(__name__)


class Model:
    def __init__(self):
        self._model = None
        self._scaler = None

    def load(self, model_dump, scaler_dump=None):
        model_dump_path = pathlib.Path(model_dump)

        if model_dump_path.suffix == ".h5":
            self._model = tf.keras.models.load_model(model_dump)
        elif model_dump_path.suffix == ".joblib":
            self._model = joblib.load(model_dump)
        else:
            raise ValueError(
                f"Unsupported model dump format '{model_dump_path.suffix}'."
            )

        log.debug("Loaded model of type '%s'.", type(self._model))

        if scaler_dump:
            scaler_dump_path = pathlib.Path(scaler_dump)

            if scaler_dump_path.suffix == ".joblib":
                self._scaler = joblib.load(scaler_dump_path)
            else:
                raise ValueError(
                    f"Unsupported scaler dump format '{scaler_dump_path.suffix}'."
                )

        log.debug("Loaded scaler of type '%s'.", type(self._scaler))

    def predict(self, x):
        if not isinstance(x, np.ndarray):
            log.debug("Input is of type '%s', converting to 'numpy.ndarray'.", type(x))
            x = np.array(x)

        # scaler expects 2D inputs
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        x_scaled = self._scaler.transform(x)

        # TODO: drop assumptions on underlying model input shape.
        # Currently assuming the underlying model is an LSTM expecting 3D
        # inputs.
        x_scaled = x_scaled.reshape(
            -1,
            *x_scaled.shape,
        )

        log.debug("x_scaled (shape=%s):\n%s", str(x_scaled.shape), str(x_scaled))

        y_scaled = self._model.predict(x_scaled)

        log.debug("y_scaled (shape=%s):\n%s", str(y_scaled.shape), str(y_scaled))

        # NOTE: assuming the expected output of the model is a single value
        return float(self._scaler.inverse_transform(y_scaled).flatten())


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
