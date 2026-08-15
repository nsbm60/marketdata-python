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
DISC_PUBSUB_FIELD = "pubSub"
DISC_ROUTER_FIELD = "router"

# Ping/pong JSON (pre-serialized for speed)
PING_JSON = '{"op":"__ping__"}'
PONG_JSON = '{"op":"__pong__"}'

# Timing defaults
HEARTBEAT_INTERVAL_SEC = 10
SELF_PING_INTERVAL_SEC = 30
SELF_PING_MAX_MISSED = 6
SELF_PING_TIMEOUT_MS = 5000
