"""
    Monasca API client
"""

import logging

from keystoneauth1.exceptions import base as keystoneauth_exception
from monascaclient import client
from monasca_agent.common import keystone

log = logging.getLogger(__name__)


class MonascaAPI(object):
    """Interface to Monasca API."""

    def __init__(self, config):
        """Initialize Monasca API client connection."""
        self._config = config
        self._mon_client = None
        self._api_version = "2_0"
        self.api_call_timeout = int(config["api_call_timeout"])

    def get_measurements(
        self,
        metric,
        instance,
        start_time,
        tenant,
        # TODO: add statistics, period, group_by, merge
    ):

        if not self._mon_client:
            self._mon_client = self._get_mon_client()
            if not self._mon_client:
                log.warning("Keystone API is down or unreachable")
                return

        # NOTE: params name must match the one expected by Monasca API
        kwargs = {
            "dimensions": {
                "resource_id": instance,
            },
            "name": metric,
            "start_time": start_time,
            "tenant_id": tenant,
        }

        return self._mon_client.metrics.list_measurements(**kwargs)

    def _get_mon_client(self):
        """Initialize Monasca API client instance with session."""
        try:
            keystone_client = keystone.Keystone(self._config)
            endpoint = keystone_client.get_monasca_url()
            session = keystone_client.get_session()
            monasca_client = client.Client(
                api_version=self._api_version,
                endpoint=endpoint,
                session=session,
                timeout=self.api_call_timeout,
                **keystone.get_args(self._config)
            )
            return monasca_client
        except keystoneauth_exception.ClientException as ex:
            log.error("Failed to initialize Monasca client. %s", ex)
