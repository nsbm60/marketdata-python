#!/usr/bin/env python3
import zmq
import json
from datetime import datetime

# --- CONFIG -------------------------------------------------
# Point this at the IB pub endpoint (same as IBPublisher bindAddress)
# For now just hard-code what IBServiceMain advertises, e.g.:
IB_PUB_ENDPOINT = "tcp://192.168.1.184:6010"  # adjust if needed
SUB_PREFIX = b"ib."                          # subscribe to ALL ib.* topics
# ------------------------------------------------------------

def main():
  ctx = zmq.Context.instance()
  sock = ctx.socket(zmq.SUB)
  sock.connect(IB_PUB_ENDPOINT)
  sock.setsockopt(zmq.SUBSCRIBE, SUB_PREFIX)

  print(f"[ib_monitor] Connected to {IB_PUB_ENDPOINT}, subscribed to {SUB_PREFIX!r}")

  try:
    while True:
      topic = sock.recv_string()          # e.g. "ib.order.DU333427"
      payload = sock.recv_string()        # JSON string
      ts = datetime.utcnow().isoformat() + "Z"

      print(f"\n[{ts}] topic={topic}")
      try:
        obj = json.loads(payload)
        print(json.dumps(obj, indent=2, sort_keys=True))
      except Exception:
        print(payload)
  except KeyboardInterrupt:
    print("\n[ib_monitor] Stopping…")
  finally:
    sock.close()
    ctx.term()

if __name__ == "__main__":
  main()
