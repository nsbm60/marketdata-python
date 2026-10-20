"""Greeks server entry point:  python -m greeks_server   (greeks-server-spec.md §9).

Startup order (P18):
  1. Load config.
  2. Resolve the bus and MDS, connect ClickHouse, fetch the calendar from MDS (waits for MDS).
  3. If in session, load SOFR and dividends.
  4. Start the compute thread and its ResilientSub (it waits for the PUB before publishing).
  5. Service.run(): connects the PUB, binds the router, then advertises. Blocks.

Exits non-zero if the compute loop dies (e.g. session inputs can't be loaded); systemd restarts
the unit and this same path runs again.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone

from discovery import ControlService, ServiceLocator
from greeks.config import load_config
from greeks.topic_builder import TopicBuilder
from greeks_server.calendar import ET, SessionCache, mds_fetch
from greeks_server.handler import OP, make_handler
from greeks_server.inputs import InputsHolder, clickhouse_loader
from greeks_server.loop import ComputeLoop
from service import Service
from service.resilient_sub import ResilientSub

log = logging.getLogger("greeks_server")

MDS_HEARTBEAT = TopicBuilder.for_service_heartbeat("mds")


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s [%(threadName)s] %(name)s - %(message)s",
    )
    cfg = load_config()
    s = cfg.server
    log.info(
        "greeks server: methodology=%s router_port=%d topic_prefix=%r clock_tick_s=%s min_pass_interval_ms=%d",
        cfg.methodology_version, s.router_port, s.topic_prefix, s.clock_tick_s, s.min_pass_interval_ms,
    )

    bus = ServiceLocator.bus()
    mds = ServiceLocator.control(ControlService.MARKET_DATA)  # waits with retry
    from ml.shared.clickhouse import get_ch_client  # canonical connector

    ch = get_ch_client()  # connect now: a bad env file fails at deploy, not at 09:30
    inputs = InputsHolder(clickhouse_loader(lambda: ch))
    now = datetime.now(timezone.utc)
    calendar = SessionCache(mds_fetch(mds.router), now.astimezone(ET).date())
    session = calendar.session_at(now)
    if session is not None:
        inputs.load_session(session.date)

    service = Service(name=ControlService.GREEKS, router_port=s.router_port)
    service.register_handler(OP, make_handler(cfg, inputs, calendar.session_at))

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
            ],
            heartbeat_topic=MDS_HEARTBEAT,
            recv_timeout_s=s.min_pass_interval_ms / 1000.0,
        )

    loop = ComputeLoop(
        cfg,
        make_subscriber=make_sub,
        publish=service.publish,
        publisher_ready=service.publisher_ready,
        inputs=inputs,
        on_fatal=on_fatal,
        session_fn=calendar.session_at,
    )
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
