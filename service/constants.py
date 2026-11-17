"""
Wire-format constants matching Scala infrastructure.

All infrastructure-level strings live here. Library code uses constants only,
never inline literals. A wire-format change is one edit, not many.
"""

import uuid

# Operation discriminators
OP_FIELD = "op"
OP_PING = "__ping__"
OP_PONG = "__pong__"

# This process's instance id, generated once at startup and carried in every pong so a client's
# ResilientDealer can tell a restart (a new process, a new id) from a steady link. Matches the Scala
# ResilientRouter, which stamps its pong the same way.
INSTANCE_FIELD = "instance"
INSTANCE_ID = str(uuid.uuid4())

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

# Ping/pong JSON (pre-serialized for speed). The pong carries this process's instance id;
# PONG_JSON (no instance) is the legacy shape a not-yet-upgraded server answers, still accepted
# as a pong by clients (they just cannot detect that server's restarts).
PING_JSON = '{"op":"__ping__"}'
PONG_JSON = '{"op":"__pong__"}'
PONG_WITH_INSTANCE = f'{{"{OP_FIELD}":"{OP_PONG}","{INSTANCE_FIELD}":"{INSTANCE_ID}"}}'

# Timing defaults
HEARTBEAT_INTERVAL_SEC = 10
SELF_PING_INTERVAL_SEC = 30
SELF_PING_MAX_MISSED = 6
SELF_PING_TIMEOUT_MS = 5000
