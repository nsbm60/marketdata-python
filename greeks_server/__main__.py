"""Greeks server entry point:  python -m greeks_server   (greeks-server-spec.md §9).

Startup order (P18):
  1. Load config.
  2. Resolve the bus and MDS, connect ClickHouse.
  3. Warm the rate/dividend inputs for today (fail fast if SOFR/ClickHouse is broken).
  4. Register the two control ops (compute_greeks, get_greeks).
  5. Start the compute thread and its ResilientSub (it waits for the PUB before publishing).
  6. Service.run(): connects the PUB, binds the router, then advertises. Blocks.

There is no calendar and no session: pricing is driven by the data (option ticks) and by MDS's
cal.options.time 5-second tick. The cold path (get_greeks for a contract the book has never seen)
asks MDS, so the MDS control handle is kept.

Exits non-zero if the compute loop dies; systemd restarts the unit and this same path runs again.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

from control_client import ControlSocket
from discovery import ControlService, ServiceLocator
from greeks.config import load_config
from greeks.topic_builder import TopicBuilder
from greeks_server.coldpath import ColdPath
from greeks_server.handler import OP, OP_GET, make_get_handler, make_handler
from greeks_server.inputs import InputsHolder, clickhouse_loader
from greeks_server.loop import ComputeLoop, methodology
from service import Service
from service.resilient_sub import ResilientSub

log = logging.getLogger("greeks_server")

ET = ZoneInfo("America/New_York")
MDS_HEARTBEAT = TopicBuilder.for_service_heartbeat("mds")
COLD_PATH_TIMEOUT_MS = 5000


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s [%(threadName)s] %(name)s - %(message)s",
    )
    cfg = load_config()
    s = cfg.server
    log.info(
        "greeks server: methodology=%s router_port=%d topic_prefix=%r",
        cfg.methodology_version, s.router_port, s.topic_prefix,
    )

    bus = ServiceLocator.bus()
    mds = ServiceLocator.control(ControlService.MARKET_DATA)  # waits with retry
    from ml.shared.clickhouse import get_ch_client  # canonical connector

    ch = get_ch_client()  # connect now: a bad env file fails at deploy, not at 09:30
    inputs = InputsHolder(clickhouse_loader(lambda: ch))
    today = datetime.now(timezone.utc).astimezone(ET).date()
    inputs.for_date(today)  # warm + fail fast if SOFR/ClickHouse is broken

    def cold_transport(req: dict[str, Any]) -> dict[str, Any]:
        sock = ControlSocket(mds.router, COLD_PATH_TIMEOUT_MS)
        try:
            return sock.request(req)
        finally:
            sock.close()

    cold = ColdPath(cold_transport, inputs, methodology(cfg))

    # Concurrent dispatch: a re-lease cold path can take seconds; running ops on a worker pool keeps
    # the main loop answering pings and other requests meanwhile (greeks-server-spec.md §6.1, §9).
    service = Service(name=ControlService.GREEKS, router_port=s.router_port, request_workers=4)

    fatal: list[BaseException] = []

    def on_fatal(e: BaseException) -> None:
        fatal.append(e)
        service.shutdown()

    def make_sub() -> ResilientSub:
        return ResilientSub(
            name="Greeks-TickSub",
            endpoint=bus.xpub,
            data_prefixes=[
                TopicBuilder.prefix_option_quotes(),
                TopicBuilder.prefix_option_trades(),
                TopicBuilder.prefix_equity_quotes(),
                TopicBuilder.prefix_equity_trades(),
                TopicBuilder.for_options_time(),  # the 5-second repricing pass
            ],
            heartbeat_topic=MDS_HEARTBEAT,
        )

    loop = ComputeLoop(
        cfg,
        make_subscriber=make_sub,
        publish=service.publish,
        publisher_ready=service.publisher_ready,
        inputs=inputs,
        cold=cold,
        on_fatal=on_fatal,
    )

    # get_greeks reads and writes the book, which the compute thread owns, so it is served by the
    # loop under its lock; compute_greeks is pure and needs neither.
    service.register_handler(OP, make_handler(cfg, inputs))
    service.register_handler(OP_GET, make_get_handler(loop.get_greeks, methodology(cfg)))

    def stop_loop() -> None:
        loop.stop()
        loop.join(timeout=5.0)

    # Runs inside Service cleanup BEFORE the PUB socket closes, so no pass publishes into it.
    service.add_shutdown_hook(stop_loop)
    loop.start()
    try:
        service.run()
    finally:
        stop_loop()
    if fatal:
        log.error("exiting: %s", fatal[0])
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
