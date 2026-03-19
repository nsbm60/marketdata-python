"""
MarketData service discovery module.

Provides ZMQ-based service discovery compatible with the Scala MarketData backend.
"""

from .service_locator import (
    ServiceLocator,
    ServiceEndpoint,
    DiscoverySubscriber,
    DEFAULT_DISCOVERY_PORT,
    SERVICE_TOPIC_PREFIX,
)

__all__ = [
    "ServiceLocator",
    "ServiceEndpoint",
    "DiscoverySubscriber",
    "DEFAULT_DISCOVERY_PORT",
    "SERVICE_TOPIC_PREFIX",
]
