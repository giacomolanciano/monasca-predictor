"""
    monasca-predictor main loop
"""

import logging
import signal
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


class PredictorProcess:
    def __init__(self):
        self._api_endpoint = None
        self._forwarder_endpoint = None
        self.run_forever = True

    def _handle_sigterm(self, signum, frame):
        log.debug("Caught SIGTERM")
        self.stop(0)

    def stop(self, exit_code):
        log.info("Stopping predictor run loop...")
        self.run_forever = False
        sys.exit(exit_code)

    def run(self, config):
        # Gracefully exit on sigterm.
        signal.signal(signal.SIGTERM, self._handle_sigterm)

        # Handle Keyboard Interrupt
        signal.signal(signal.SIGINT, self._handle_sigterm)

        if not self._api_endpoint:
            self._api_endpoint = MonascaAPI(config)

        inference_frequency = config.get("inference_frequency_seconds")

        # NOTE: take start time here so that absolute timing can be used and
        # cumulative delays mitigated.
        inference_start = time.time()

        while self.run_forever:
            now = datetime.utcfromtimestamp(inference_start)

            for instance in instance_list:
                instance_id = instance.get("id")
                tenant_id = instance.get("tenant_id")
                start_time = util.format_datetime_str(
                    now - timedelta(seconds=instance.get("lookback_period_seconds"))
                )
                group_by = instance.get("group_by")
                merge_metrics = instance.get("merge_metrics")
                statistics = instance.get("statistics")
                aggregation_period_seconds = instance.get("aggregation_period_seconds")

                # TODO: handle measurements coming from multiple input metrics
                in_metric_list = instance.get("metrics")

                if not in_metric_list:
                    log.info(
                        "No input metrics for instance '%s'. Skipping...",
                        instance_id,
                    )
                    continue

                for metric in in_metric_list:
                    log.info(
                        "Getting '%s' measurements for instance '%s'...",
                        metric,
                        instance_id,
                    )

                    get_measurement_resp = self._api_endpoint.get_measurements(
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
                    measurement_list = get_measurement_resp[0]

                    log.debug(
                        "Received response containing the following measurements:\n%s",
                        util.format_object_str(measurement_list),
                    )

                # TODO: feed predictor with measurements
                out_metric_name = f"pred.{in_metric_list[0]}"
                predictor_value = 0.0

                # NOTE: associate predictor output with last input measurement
                # timestamp. Having a temporal relation between the two is
                # useful for plotting and time-series analysis in general. The
                # (future) timestamp that the prediction refer to will be
                # included in the dimensions.
                #
                # Consider that, if the *inference* frequency is lower than the
                # *collection* frequency (30 secs, by default), some predictor
                # outputs will be overridden, as the last input measurement
                # timestamp (actually, the whole input) will be the same for
                # multiple consecutive inference runs. However, setting an
                # inference frequency that low makes sense only for debugging.
                last_datetime_str = measurement_list["measurements"][-1][0]
                predictor_datetime = util.get_parsed_datetime(last_datetime_str)
                predictor_timestamp = predictor_datetime.timestamp()

                log.debug("Last input measurement datetime: %s", last_datetime_str)
                log.debug("Predictor output datetime: %s", str(predictor_datetime))
                log.debug("Predictor output timestamp: %f", predictor_timestamp)

                log.info(
                    "Sending '%s' measurement for instance '%s' to forwarder...",
                    out_metric_name,
                    instance_id,
                )

                self._send_to_forwarder(
                    out_metric_name,
                    predictor_value,
                    predictor_timestamp,
                    measurement_list["dimensions"],
                    tenant_id=tenant_id,
                    forwarder_endpoint=config["forwarder_url"],
                )

            # Only plan for the next loop if we will continue,
            # otherwise just exit quickly.
            if self.run_forever:
                inference_elapsed_time = time.time() - inference_start
                if inference_elapsed_time < inference_frequency:
                    # TODO: something like C's clock_nanosleep() should be
                    # used, instead of sleep(), to have a more precise timing
                    # and be more robust to cumulative delays
                    time.sleep(inference_frequency - inference_elapsed_time)

                    inference_start += inference_frequency
                else:
                    log.info(
                        "Inference took %f which is as long or longer then the configured "
                        "inference frequency of %d. Starting inference again without waiting "
                        "in result...",
                        inference_elapsed_time,
                        inference_frequency,
                    )
        self.stop(0)

    @staticmethod
    def _send_to_forwarder(
        metric_name, value, timestamp, dimensions, tenant_id, forwarder_endpoint
    ):
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

        http_emitter([envelope], log, forwarder_endpoint)


def main():
    options, args = util.get_parsed_args()
    predictor_config = PredictorConfig().get_config(["Main", "Api", "Logging"])

    log.debug(
        "monasca-predictor started with the following configs:\n%s",
        util.format_object_str(predictor_config),
    )

    global instance_list
    instance_list = predictor_config["instances"]

    predictor = PredictorProcess()
    predictor.run(predictor_config)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        try:
            log.exception("Uncaught error during monasca-predictor execution.")
        except Exception:  # pylint: disable=broad-except
            pass
        raise
