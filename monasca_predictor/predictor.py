"""
    monasca-predictor main loop
"""

import logging
import sys

import monasca_agent.common.util as agent_util
from monasca_predictor.common.config import PredictorConfig

log = logging.getLogger(__name__)


def main():
    options, args = agent_util.get_parsed_args(prog="monasca-predictor")
    predictor_config = PredictorConfig().get_config(["Main", "Api", "Logging"])

    log.debug("predictor configs: %s", str(predictor_config))

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        try:
            log.exception("Uncaught error during monasca-predictor execution")
        except Exception:  # pylint: disable=broad-except
            pass
        raise
