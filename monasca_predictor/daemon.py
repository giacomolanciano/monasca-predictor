"""
    monasca-predictor main loop
"""

import copy
import logging
import signal
import sys
import time
from datetime import timedelta, datetime

import monasca_agent.common.metrics as metrics
from monasca_agent.common.emitter import http_emitter

import monasca_predictor.common.util as util
import monasca_predictor.inference.model as inf_mod
from monasca_predictor.api.monasca_api import MonascaAPI
from monasca_predictor.common.config import PredictorConfig


log = logging.getLogger(__name__)
prediction_config_list = []


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

            for prediction_id, prediction_config in enumerate(prediction_config_list):
                in_metric_list = prediction_config.get("metrics")

                if not in_metric_list:
                    log.info(
                        "No input metrics specified for prediction %d, skipping...",
                        prediction_id,
                    )
                    continue

                # Get config parameters
                tenant_id = prediction_config.get("tenant_id")
                instance_id = prediction_config.get("instance_id")
                dimensions = prediction_config.get("dimensions")

                lookback_period_seconds = prediction_config.get(
                    "lookback_period_seconds"
                )
                prediction_offset_seconds = prediction_config.get(
                    "prediction_offset_seconds"
                )

                statistics = prediction_config.get("statistics")
                aggregation_period_seconds = prediction_config.get(
                    "aggregation_period_seconds"
                )

                group_by = prediction_config.get("group_by")
                merge_metrics = prediction_config.get("merge_metrics")

                start_time = util.format_datetime_str(
                    now - timedelta(seconds=lookback_period_seconds)
                )
                expected_input_shape = (
                    lookback_period_seconds // aggregation_period_seconds
                )

                # Fetch measurements (by metric) from Monasca API.
                #
                # NOTE: The result of this block is a dict indexed by instance
                # ID. The value of a single entry is a dict containing a list
                # of lists of measurements ("measurements") and the dimensions
                # of the instance ("dimensions").
                instance_metrics_dict = {}

                for metric in in_metric_list:
                    log.info(
                        "Getting '%s' measurements for prediction %d...",
                        metric,
                        prediction_id,
                    )

                    get_measurement_resp = self._api_endpoint.get_measurements(
                        metric=metric,
                        start_time=start_time,
                        instance_id=instance_id,
                        tenant=tenant_id,
                        dimensions=dimensions,
                        group_by=group_by,
                        merge_metrics=merge_metrics,
                        statistics=statistics,
                        aggregation_period_seconds=aggregation_period_seconds,
                    )

                    log.debug(
                        "Received the following response:\n%s",
                        util.format_object_str(get_measurement_resp),
                    )

                    for instance in get_measurement_resp:
                        instance_id = instance["dimensions"]["resource_id"]

                        if statistics:
                            measurement_list = instance["statistics"]
                        else:
                            measurement_list = instance["measurements"]

                        # NOTE: value list must be checked such that it matches the
                        # expected input shape of the model. It could be the case that
                        # the retrieved metrics are too few (because the collection has
                        # just started and the lookback period is too long) or too
                        # much (because of the effect of cumulative delays in the
                        # collection process).
                        if (
                            not measurement_list
                            or len(measurement_list) < expected_input_shape
                        ):

                            log.info(
                                "Retrieved measurements are too few "
                                "(expected: %d, actual: %d). Skipping...",
                                expected_input_shape,
                                len(measurement_list),
                            )
                            continue

                        elif len(measurement_list) > expected_input_shape:

                            log.info(
                                "Retrieved measurements are too much, retaining only last %d.",
                                expected_input_shape,
                            )

                            measurement_list = measurement_list[-expected_input_shape:]

                        log.debug(
                            "Value list for metric '%s' of instance '%s' is:\n%s",
                            metric,
                            instance_id,
                            measurement_list,
                        )

                        # Update instance entry
                        if instance_metrics_dict.get(instance_id):
                            instance_metrics_dict[instance_id]["measurements"].append(
                                measurement_list
                            )
                        else:
                            instance_metrics_dict[instance_id] = {
                                "dimensions": instance["dimensions"],
                                "measurements": [measurement_list],
                            }

                log.debug(
                    "Retrieve following data from Monasca API:\n%s",
                    util.format_object_str(instance_metrics_dict),
                )

                # Feed predictor with retrieved measurements
                model_type = prediction_config.get("model_type")
                if model_type and model_type == "linear":
                    model = inf_mod.LinearModel(
                        prediction_offset_seconds / aggregation_period_seconds
                    )
                else:
                    model = inf_mod.Model()
                    model.load(
                        model_dump=prediction_config.get("model_path"),
                        scaler_dump=prediction_config.get("scaler_path"),
                    )

                for instance_id, instance_data in instance_metrics_dict.items():

                    # TODO: handle measurements coming from multiple input metrics.
                    # Currently assuming only one input metric is specified.
                    measurement_list = instance_data["measurements"][0]
                    measurement_value_list = [x[1] for x in measurement_list]
                    prediction_value = model.predict(measurement_value_list)

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
                    last_datetime_str = measurement_list[-1][0]
                    prediction_datetime = util.get_parsed_datetime(last_datetime_str)
                    prediction_timestamp = prediction_datetime.timestamp()

                    log.debug("Last input measurement datetime: %s", last_datetime_str)
                    log.debug("Predictor output datetime: %s", str(prediction_datetime))
                    log.debug("Predictor output timestamp: %f", prediction_timestamp)

                    # Build predictor output dimensions
                    prediction_dimensions = copy.deepcopy(instance_data["dimensions"])

                    # TODO restore future timestamp
                    # When providing future_timestamp as additional dimension,
                    # Monasca seems not to be able to understand that the
                    # measurements belong to the same metric. This behavior
                    # makes impossible to use the predicted metric for
                    # autoscaling purposes
                    #
                    # prediction_future_datetime =
                    # util.format_datetime_str(prediction_datetime +
                    # timedelta(seconds=prediction_offset_seconds)
                    # )
                    # prediction_dimensions["future_timestamp"] =
                    #     prediction_future_datetime

                    out_metric_name = f"pred.{in_metric_list[0]}"

                    log.info(
                        "Sending '%s' measurement for instance '%s' to forwarder...",
                        out_metric_name,
                        instance_id,
                    )

                    self._send_to_forwarder(
                        out_metric_name,
                        prediction_value,
                        prediction_timestamp,
                        prediction_dimensions,
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

    global prediction_config_list
    prediction_config_list = predictor_config["predictions"]

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
