[![GitHub license](https://img.shields.io/github/license/giacomolanciano/monasca-predictor)](https://github.com/giacomolanciano/monasca-predictor/blob/master/LICENSE)
[![DOI](https://zenodo.org/badge/343854707.svg)](https://zenodo.org/badge/latestdoi/343854707)

# monasca-predictor

`monasca-predictor` is a Python module that integrates with [Monasca](https://docs.openstack.org/monasca) to provide
OpenStack services with predictive analytics based on the collected measurements.

For instance, `monasca-predictor` can be used to leverage on time-series forecasting techniques and enable a predictive
auto-scaling strategy. The following figure shows how this component can be integrated into a standard OpenStack
elasticity control-loop to automatically resize a Senlin cluster of Nova instances:

<p align="center"><img src="docs/img/pred-auto-scaling.png" alt="pred-auto-scaling" width="80%"/></p>

In this case, depending on the future status of the system (as depicted by the relevant predictive analytics), one can
configure Monasca alarms to trigger appropriate Senlin scaling actions to reconfigure the cluster.

## Installation

This project is compatible with Python `3.7.*`. Assuming `venv` to be available on the system, run the following
commands to install:

```bash
git clone https://github.com/giacomolanciano/monasca-predictor
cd monasca-predictor
make py37
```

The `monasca-predictor` command can now be issued from within the newly created virtual env, that can be activated by
running

```bash
source .venv/py37/bin/activate
```

## Configuration

This component assumes OpenStack, and Monasca in particular, to be already installed on the system and configured
appropriately.

Similarly to Monasca components, it is possible to configure the behavior of `monasca-predictor` through a `yaml` config
file structured as follows (default values specified for optional fields only):

```yaml
Api:
  ###################################################################
  # This section allows for specifying the information required to  #
  # call OpenStack APIs. It is recommended to fill in the fields    #
  # using the same values that can be found in the config file of   #
  # monasca-agent-forwarder.                                        #
  ###################################################################
  ca_file: ...
  endpoint_type: ...
  insecure: ...
  keystone_url: ...
  password: ...
  project_domain_id: ...
  project_domain_name: ...
  project_name: ...
  region_name: ...
  service_type: ...
  user_domain_name: ...
  username: ...
Logging:
  # Whether to enable log rotation
  enable_logrotate: true

  # Whether to disable logging to file
  disable_file_logging: false

  # Path to log file
  predictor_log_file: /var/log/monasca-predictor/predictor.log

  # The minimum severity level of messages to be logged
  log_level: INFO
Main:
  ###################################################################
  # The following parameters allows for specifying the information  #
  # required to call the monasca-agent-forwarder endpoint. It is    #
  # recommended to fill in the fields using the same values that    #
  # can be found in the config file of monasca-agent-collector.     #
  forwarder_url: ...
  hostname: ...
  ###################################################################

  # The time interval (in seconds) to wait between predictor activations
  inference_frequency_seconds: ...

  # A list of predictive analytics to be produced
  predictions:
      # The ID of the OpenStack project containing the objects to be monitored
    - tenant_id:

      # A map of custom properties that the objects to be monitored must match
      dimensions:
        key: value
        ...

      # A list of metrics to be retrieved for the object to be monitored
      metrics: [ ... ]

      # The name of the output (predictive) metric
      out_metric: ...

      # A list of statistics to compute temporal aggregations of the input metrics with
      # (e.g., avg, max, cnt)
      time_aggregation_statistics: []

      # The time interval (in seconds) to consider when computing temporal aggregations
      # (required only when time_aggregation_statistics is non-empty)
      time_aggregation_period_seconds: ...

      # A list of statistics to compute spatial aggregations of the input metrics with
      # (e.g., avg, max, cnt)
      space_aggregation_statistics: []

      # A list of properties to use for grouping (a '*' groups by all keys)
      group_by: []

      # Whether to merge metrics coming from different objects into a single series
      merge_metrics: false

      # The forecasting time period (in seconds, depends on the underlying model)
      prediction_offset_seconds: ...

      # The amount of historical data to look at for a single prediction (in seconds,
      # depends on the underlying model)
      lookback_period_seconds: 1200

      # The standard type of forecasting model to use (only 'linear' is supported).
      # This value takes precedence over model_path.
      model_type: ...

      # Path to forecasting model dump (.h5, .pt, .joblib). This value is ignored
      # when model_type is specified.
      model_path: ...

      # Path to data scaler dump (.joblib)
      scaler_path: ...

    - ...
```

## Run

Assuming you are within the expected virtual env (see [Installation](#installation)) and prepared a config file named `predictor.yaml` (see [Configuration](#configuration)), you can launch `monasca-predictor` by running

```bash
monasca-predictor -f predictor.yaml
```

**NOTE:** in general, you may need `monasca-predictor` to run in the background for a relatively long period. Therefore,
it is recommended to start it from within a `screen`, or whatever method you prefer to run processes in detached mode.

## Getting started

If you want to see `monasca-predictor` in action, you can check out the paper *"Predictive Auto-Scaling with OpenStack
Monasca"* (accepted at UCC 2021) and its [companion
repo](https://github.com/giacomolanciano/UCC2021-predictive-auto-scaling-openstack), including: OpenStack quick setup
instructions, example config files, results from the paper, etc.

If you found `monasca-predictor` (and the aforementioned material) useful for your research work, please consider
citing:

```bibtex
@inproceedings{Lanciano2021Predictive,
  author={Lanciano, Giacomo and Galli, Filippo and Cucinotta, Tommaso and Bacciu, Davide and Passarella, Andrea},
  booktitle={2021 IEEE/ACM 14th International Conference on Utility and Cloud Computing (UCC)},
  title={Predictive Auto-scaling with OpenStack Monasca},
  year={2021},
  doi={10.1145/3468737.3494104},
}
```

## Contribution

All kind of contributions (e.g., bug reports, bug fixes, new feature ideas, documentation enhancements) are welcome.
Feel free to open a new [issue](https://github.com/giacomolanciano/monasca-predictor/issues) to get in touch with the
maintainers.

## License

This project is licensed under the terms of [Apache 2.0 license](LICENSE).
