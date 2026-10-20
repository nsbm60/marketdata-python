"""
The closed set of services discovery knows, grouped by advertisement shape, and the typed result
each shape resolves to. Mirrors the Scala discovery/Services.scala.

ServiceLocator is exhaustive by knowledge: it knows every service and the shape it advertises. An
unknown service — a non-member argument — raises at the call, the runtime equivalent of the Scala
compile error.

Wire names are the values that already travel on the wire (matching the Scala enums), so nothing on
the wire changes. Discovery topics are subscribed by prefix and lowercased, so case does not matter.
"""

from dataclasses import dataclass
from enum import Enum


class ControlService(Enum):
    """A service with a control plane: it advertises a control router."""
    MARKET_DATA = "marketData"
    CALC = "calc"
    IB = "ib"
    ALPACA_TRADING = "alpaca"
    OPTIMIZER = "optimizer"
    # Python-only: the service-infra test harness (tools/run_echo_service.py). No Scala counterpart.
    ECHO_TEST = "echo_test"


class EndpointService(Enum):
    """A pure endpoint: it advertises a host and port, no control plane."""
    CLICKHOUSE = "clickhouse"
    IB_GATEWAY = "ibgateway"


# The pub/sub bus. It advertises its two data-plane endpoints and nothing else.
PUBSUB_BUS = "pubsubbus"


@dataclass(frozen=True)
class ControlEndpoint:
    """A control service's control endpoint. No data field: the bus is the only data plane."""
    host: str
    router: str


@dataclass(frozen=True)
class HostPort:
    """A pure endpoint's host and port."""
    host: str
    port: int


@dataclass(frozen=True)
class BusEndpoints:
    """The pub/sub bus's two data-plane endpoints (it has no control plane)."""
    xsub: str   # producers connect in
    xpub: str   # subscribers connect out
