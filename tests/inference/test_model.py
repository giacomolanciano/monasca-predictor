import logging
import pathlib

from monasca_predictor.inference.model import Model

log = logging.getLogger(__name__)
tests_dir = pathlib.Path(__file__).parent.absolute()


def test_model():
    model_dump = (tests_dir / "model.h5").absolute()
    scaler_dump = (tests_dir / "scaler.joblib").absolute()

    log.debug("Using model dump at '%s'.", model_dump)
    log.debug("Using scaler dump at '%s'.", scaler_dump)

    model = Model()
    model.load(
        model_dump=model_dump,
        scaler_dump=scaler_dump,
    )

    x = [0.0] * 24
    y = model.predict(x)

    log.debug("y (%s): %s", type(y), str(y))


if __name__ == "__main__":
    test_model()
