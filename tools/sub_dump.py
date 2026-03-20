#!/usr/bin/env python3
import zmq, time, sys

# Default ports: 6006=MDS, 6020=CalcServer
port = int(sys.argv[1]) if len(sys.argv) > 1 else 6006
PUB_ENDPOINT = f"tcp://127.0.0.1:{port}"

ctx = zmq.Context.instance()
sub = ctx.socket(zmq.SUB)
sub.connect(PUB_ENDPOINT)

# Subscribe to everything (or set a prefix like "MD." if you know it)
sub.setsockopt_string(zmq.SUBSCRIBE, "")

print(f"Listening on {PUB_ENDPOINT} for 20s…")
end = time.time() + 20
while time.time() < end:
    try:
        topic = sub.recv_string(flags=zmq.NOBLOCK)  # frame 1
        payload = sub.recv_string()                 # frame 2
        print(f"[{topic}] {payload}")
    except zmq.Again:
        time.sleep(0.05)
