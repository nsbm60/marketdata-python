"""
Wire-format constants matching Scala infrastructure.

All infrastructure-level strings live here. Library code uses constants only,
never inline literals. A wire-format change is one edit, not many.
"""

# Operation discriminators
OP_FIELD = "op"
OP_PING = "__ping__"
OP_PONG = "__pong__"

# Response envelope fields
OK_FIELD = "ok"
DATA_FIELD = "data"
ERROR_FIELD = "error"

# Heartbeat
HEARTBEAT_TOPIC_SUFFIX = "heartbeat"
HEARTBEAT_TS_FIELD = "ts"

# Discovery announcement (UDP broadcast)
DISCOVERY_BROADCAST_ADDRESS = "255.255.255.255"
DISCOVERY_PORT = 6001
DISCOVERY_INTERVAL_SEC = 5
DISC_SERVICE_FIELD = "service"
DISC_HOST_FIELD = "host"
DISC_ROUTER_FIELD = "router"
# Pub/sub bus endpoints (additive). The bus is a two-port XSUB/XPUB proxy with no
# control plane, so it advertises these EXPLICITLY instead of overloading `router`
# to carry XSUB. Absent for every other service.
DISC_XSUB_FIELD = "xsub"   # producers connect in
DISC_XPUB_FIELD = "xpub"   # subscribers connect out

# Ping/pong JSON (pre-serialized for speed)
PING_JSON = '{"op":"__ping__"}'
PONG_JSON = '{"op":"__pong__"}'

# Timing defaults
HEARTBEAT_INTERVAL_SEC = 10
SELF_PING_INTERVAL_SEC = 30
SELF_PING_MAX_MISSED = 6
SELF_PING_TIMEOUT_MS = 5000
