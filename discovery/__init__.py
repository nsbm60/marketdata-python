"""
MarketData service discovery module.

Provides ZMQ-based service discovery compatible with the Scala MarketData backend.
"""

from .service_locator import (
    ServiceLocator,
    DiscoverySubscriber,
    DEFAULT_DISCOVERY_PORT,
    SERVICE_TOPIC_PREFIX,
)
from .services import (
    ControlService,
    EndpointService,
    PUBSUB_BUS,
    ControlEndpoint,
    HostPort,
    BusEndpoints,
)

__all__ = [
    "ServiceLocator",
    "DiscoverySubscriber",
    "DEFAULT_DISCOVERY_PORT",
    "SERVICE_TOPIC_PREFIX",
    "ControlService",
    "EndpointService",
    "PUBSUB_BUS",
    "ControlEndpoint",
    "HostPort",
    "BusEndpoints",
]
