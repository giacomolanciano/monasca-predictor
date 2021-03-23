"""
    monasca-predictor main loop
"""

import json
import logging
import sys
import time
from datetime import timedelta, datetime

import monasca_agent.common.metrics as metrics
from monasca_agent.common.emitter import http_emitter

import monasca_predictor.common.util as util
from monasca_predictor.api.monasca_api import MonascaAPI
from monasca_predictor.common.config import PredictorConfig


log = logging.getLogger(__name__)
instance_list = []


class PredictorDaemon:
    def __init__(self, config):
        self._api_endpoint = MonascaAPI(config)
        self._forwarder_endpoint = config["forwarder_url"]

    def run(self):
        now = datetime.utcnow()
        now_str = util.format_datetime_str(now)

        for instance in instance_list:
            instance_id = instance["id"]
            tenant_id = instance["tenant_id"]
            start_time = util.format_datetime_str(
                now - timedelta(minutes=instance["lookback_period_minutes"])
            )

            log.debug("instance id: %s", instance_id)
            log.debug("tenant id: %s", tenant_id)
            log.debug("lookback period start: %s", start_time)

            # TODO: handle measurements coming from multiple input metrics
            in_metric_list = instance["metrics"]
            for metric in in_metric_list:
                log.info(
                    "Requesting '%s' measurements, from %s to %s, for instance '%s' ...",
                    metric,
                    start_time,
                    now_str,
                    instance_id,
                )

                measurements = self._api_endpoint.get_measurements(
                    metric=metric,
                    instance=instance_id,
                    start_time=start_time,
                    tenant=tenant_id,
                )

                # NOTE: unpack measurements, assuming they come from a single instance
                measurements = measurements[0]

                log.debug("measurements:")
                for line in json.dumps(measurements, indent=2).splitlines():
                    log.debug("%s", line)

            # TODO: feed predictor with measurements
            predictor_value = 0.0
            predictor_timestamp = time.time()

            out_metric_name = f"pred.{in_metric_list[0]}"
            out_metric = metrics.Metric(
                name=out_metric_name,
                dimensions=measurements["dimensions"],
                tenant=tenant_id,
            )
            envelope = out_metric.measurement(predictor_value, predictor_timestamp)

            log.debug("envelope:")
            for line in json.dumps(envelope, indent=2).splitlines():
                log.debug("%s", line)

            log.info(
                "Relaying '%s' measurement for instance %s to forwarder ...",
                out_metric_name,
                instance_id,
            )

            http_emitter([envelope], log, self._forwarder_endpoint)


def main():
    options, args = util.get_parsed_args()
    predictor_config = PredictorConfig().get_config(["Main", "Api", "Logging"])

    log.debug("predictor configs:")
    for line in json.dumps(predictor_config, indent=2).splitlines():
        log.debug("%s", line)

    global instance_list
    instance_list = predictor_config["instances"]

    predictor = PredictorDaemon(predictor_config)
    predictor.run()

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
