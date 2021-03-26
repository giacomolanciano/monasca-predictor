"""
    monasca-predictor main loop
"""

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

        for instance in instance_list:
            instance_id = instance.get("id")
            tenant_id = instance.get("tenant_id")
            start_time = util.format_timestamp_str(
                now - timedelta(minutes=instance.get("lookback_period_minutes"))
            )
            group_by = instance.get("group_by")
            merge_metrics = instance.get("merge_metrics")
            statistics = instance.get("statistics")
            aggregation_period_seconds = instance.get("aggregation_period_seconds")

            # TODO: handle measurements coming from multiple input metrics
            in_metric_list = instance["metrics"]
            for metric in in_metric_list:
                log.info(
                    "Getting '%s' measurements for instance '%s' ...",
                    metric,
                    instance_id,
                )

                measurements = self._api_endpoint.get_measurements(
                    metric=metric,
                    start_time=start_time,
                    instance=instance_id,
                    tenant=tenant_id,
                    group_by=group_by,
                    merge_metrics=merge_metrics,
                    statistics=statistics,
                    aggregation_period_seconds=aggregation_period_seconds,
                )

                # NOTE: unpack measurements, assuming they come from a single instance
                measurements = measurements[0]

                log.debug(
                    "Received response containing the following measurements:\n%s",
                    util.format_object_str(measurements),
                )

            # TODO: feed predictor with measurements
            out_metric_name = f"pred.{in_metric_list[0]}"
            predictor_value = 0.0
            predictor_timestamp = time.time()

            log.info(
                "Relaying '%s' measurement for instance %s to forwarder ...",
                out_metric_name,
                instance_id,
            )

            self._send_to_forwarder(
                out_metric_name,
                predictor_value,
                predictor_timestamp,
                measurements["dimensions"],
                tenant_id=tenant_id,
            )

    def _send_to_forwarder(self, metric_name, value, timestamp, dimensions, tenant_id):
        out_metric = metrics.Metric(
            name=metric_name,
            dimensions=dimensions,
            tenant=tenant_id,
        )
        envelope = out_metric.measurement(value, timestamp)

        log.debug(
            "The following envelope will be sent to forwarder:\n%s",
            util.format_object_str(envelope),
        )

        http_emitter([envelope], log, self._forwarder_endpoint)


def main():
    options, args = util.get_parsed_args()
    predictor_config = PredictorConfig().get_config(["Main", "Api", "Logging"])

    log.debug(
        "monasca-predictor started with the following configs:\n%s",
        util.format_object_str(predictor_config),
    )

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
