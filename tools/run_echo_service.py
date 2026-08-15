#!/usr/bin/env python3
"""
Echo service — test harness for the service infrastructure library.

Registers two handlers:
  - "echo": returns {"received": <full request payload>}
  - "fail": raises ValueError (tests exception-to-error wrapping)

Start this, then run test_service_infra.py in another terminal.
"""

import logging
import sys

sys.path.insert(0, ".")
from service import Service

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)-5s [%(threadName)s] %(name)s - %(message)s",
)


def echo_handler(payload: dict) -> dict:
    return {"received": payload}


def fail_handler(payload: dict) -> dict:
    raise ValueError("intentional test failure")


def main():
    service = Service(
        name="echo_test",
        pub_port=6090,
        router_port=6091,
    )
    service.register_handler("echo", echo_handler)
    service.register_handler("fail", fail_handler)
    service.run()


if __name__ == "__main__":
    main()
