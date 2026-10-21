#!/usr/bin/env python3
"""Shadow check for the greeks server (greeks-server-spec.md §10, step 5).

Subscribes to the vendor greeks MDS publishes on md.option.greeks.* and the local greeks the
server publishes on shadow.md.option.greeks.*, and reports:

  1. Differences, by moneyness (K/S) and DTE bucket, for contracts where both sides have a
     success: median and p90 of |dIV| (bps), |d delta|, |d gamma| / vendor gamma, |d theta|.
     The latest value on each side is compared; pairs more than --max-skew-s apart are skipped.
     Plus two clock-convention checks per bucket:
       - signed IV gap, median (IV_local - IV_vendor) in bps: a clock difference has one sign;
       - implied clock offset in hours: the vendor T that reconciles the two IVs at the same
         price, T_v = t_local * (IV_local / IV_vendor)^2 (sigma^2 T held fixed), offset =
         (T_v - t_local) * 8760. Median and p10..p90. K/S 0.90-1.10 only (tick noise outside).
       Reading it: a roughly constant offset across DTE buckets is a clock convention (about +8 h:
       vendor measures to midnight, not 16:00; about +24 h: whole days; drifting with time of
       day: vendor T doesn't move intraday). A scattered offset is a price-source difference.
  2. The cost of having no seed: how many contracts with live vendor greeks have no local success
     1, 5 and 15 minutes after a server restart. A restart is the first greeks.heartbeat after a
     gap of more than 25 s (or the first one the tool sees).

Run on any box that can reach the bus, during the options session, for several sessions:

    python tools/greeks_shadow_compare.py
    python tools/greeks_shadow_compare.py --interval 300 --jsonl data/greeks_shadow.jsonl

Units are compared as published. Ours: §2.1 of the spec. Massive: iv decimal, delta and gamma
per share, theta per calendar day, vega per vol point — the same as ours, so raw differences
are meaningful.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from discovery import ServiceLocator  # noqa: E402
from greeks.occ import parse_occ  # noqa: E402
from greeks.topic_builder import TopicBuilder  # noqa: E402
from greeks_server.calendar import ET  # noqa: E402
from service.resilient_sub import ResilientSub  # noqa: E402

log = logging.getLogger("greeks_shadow")

VENDOR = TopicBuilder.prefix_option_greeks()            # md.option.greeks.
LOCAL = "shadow." + VENDOR                               # shadow.md.option.greeks.
GREEKS_HB = TopicBuilder.for_service_heartbeat("greeks")
MDS_HB = TopicBuilder.for_service_heartbeat("mds")
RESTART_GAP_S = 25.0
COVERAGE_MARKS_S = (60, 300, 900)
VENDOR_LIVE_S = 120.0

MONEYNESS = ((0.0, 0.90, "<0.90"), (0.90, 0.97, "0.90-0.97"), (0.97, 1.03, "0.97-1.03"),
             (1.03, 1.10, "1.03-1.10"), (1.10, 99.0, ">1.10"))
DTE = ((0, 1, "0-1d"), (2, 7, "2-7d"), (8, 30, "8-30d"), (31, 60, "31-60d"), (61, 10_000, ">60d"))


@dataclass
class Greeks:
    iv: float
    delta: float
    gamma: float
    theta: float
    at: float  # receipt time, monotonic
    t_years: Optional[float] = None  # local only: the T the server used


def _f(d: dict[str, Any], k: str) -> Optional[float]:
    v = d.get(k)
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def parse_vendor(payload: str, at: float) -> Optional[tuple[str, Greeks]]:
    d = json.loads(payload)
    g = d.get("greeks") or {}
    vals = [_f(g, k) for k in ("iv", "delta", "gamma", "theta")]
    if not isinstance(d.get("osi"), str) or any(v is None for v in vals):
        return None
    iv, de, ga, th = vals  # type: ignore[misc]
    return d["osi"], Greeks(iv, de, ga, th, at)  # type: ignore[arg-type]


def parse_local(payload: str, at: float) -> Optional[tuple[str, Greeks, Optional[float]]]:
    d = json.loads(payload)
    if d.get("status") != "success":
        return None
    return (
        d["osi"],
        Greeks(float(d["iv"]), float(d["delta"]), float(d["gamma"]), float(d["theta"]), at,
               t_years=_f(d, "t_years")),
        _f(d, "spot"),
    )


def _bucket(x: float, table: tuple[tuple[Any, Any, str], ...]) -> str:
    for lo, hi, name in table:
        if lo <= x <= hi if isinstance(lo, int) else lo <= x < hi:
            return name
    return "?"


def _q(xs: list[float], p: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, int(p * len(s)))]


class Shadow:
    def __init__(self, max_skew_s: float) -> None:
        self.vendor: dict[str, Greeks] = {}
        self.local: dict[str, Greeks] = {}
        self.spot: dict[str, float] = {}
        self.max_skew_s = max_skew_s
        self.last_greeks_hb: Optional[float] = None
        self.restart_at: Optional[float] = None
        self.local_since_restart: set[str] = set()
        self.marks_done: set[int] = set()

    def on(self, topic: str, payload: str, now: float) -> None:
        if topic.startswith(GREEKS_HB):
            if self.last_greeks_hb is None or now - self.last_greeks_hb > RESTART_GAP_S:
                self.restart_at = now
                self.local_since_restart = set()
                self.marks_done = set()
                log.info("greeks server (re)start detected")
            self.last_greeks_hb = now
        elif topic.startswith(LOCAL):
            r = parse_local(payload, now)
            if r is not None:
                osi, g, spot = r
                self.local[osi] = g
                self.local_since_restart.add(osi)
                if spot is not None:
                    self.spot[parse_occ(osi).root] = spot
        elif topic.startswith(VENDOR):
            r2 = parse_vendor(payload, now)
            if r2 is not None:
                self.vendor[r2[0]] = r2[1]

    def coverage(self, now: float) -> list[dict[str, Any]]:
        out = []
        if self.restart_at is None:
            return out
        for mark in COVERAGE_MARKS_S:
            if mark in self.marks_done or now - self.restart_at < mark:
                continue
            live = {o for o, g in self.vendor.items() if now - g.at <= VENDOR_LIVE_S}
            missing = live - self.local_since_restart
            out.append({"kind": "coverage", "after_s": mark, "vendor_live": len(live), "no_local": len(missing),
                        "sample": sorted(missing)[:5]})
            self.marks_done.add(mark)
        return out

    def report(self) -> dict[str, Any]:
        today = datetime.now(ET).date()
        cells: dict[tuple[str, str], dict[str, list[float]]] = {}
        for osi, v in self.vendor.items():
            l = self.local.get(osi)
            if l is None or abs(l.at - v.at) > self.max_skew_s:
                continue
            try:
                c = parse_occ(osi)
            except ValueError:
                continue
            s = self.spot.get(c.root)
            if s is None:
                continue
            ks = c.strike / s
            key = (_bucket(ks, MONEYNESS), _bucket((c.expiry - today).days, DTE))
            cell = cells.setdefault(key, {"iv_bps": [], "delta": [], "gamma_rel": [], "theta": [],
                                          "iv_signed_bps": [], "offset_h": []})
            cell["iv_bps"].append(abs(l.iv - v.iv) * 1e4)
            cell["iv_signed_bps"].append((l.iv - v.iv) * 1e4)
            if 0.90 <= ks <= 1.10 and l.iv > 0 and v.iv > 0 and l.t_years is not None and l.t_years > 0:
                t_vendor = l.t_years * (l.iv / v.iv) ** 2
                cell["offset_h"].append((t_vendor - l.t_years) * 8760.0)
            cell["delta"].append(abs(l.delta - v.delta))
            if v.gamma != 0:
                cell["gamma_rel"].append(abs(l.gamma - v.gamma) / abs(v.gamma))
            cell["theta"].append(abs(l.theta - v.theta))
        rows = []
        for (m, d), cell in sorted(cells.items()):
            row: dict[str, Any] = {"moneyness": m, "dte": d, "n": len(cell["iv_bps"])}
            for k, xs in cell.items():
                if xs:
                    row[f"{k}_med"] = statistics.median(xs)
                    row[f"{k}_p90"] = _q(xs, 0.9)
            if cell["offset_h"]:
                row["offset_h_p10"] = _q(cell["offset_h"], 0.1)
                row["n_offset"] = len(cell["offset_h"])
            rows.append(row)
        return {"kind": "diff", "vendor": len(self.vendor), "local": len(self.local), "rows": rows}


def _print_report(r: dict[str, Any]) -> None:
    print(f"\n{datetime.now(timezone.utc).isoformat()}  vendor={r['vendor']} local={r['local']}")
    print(f"{'K/S':>10} {'DTE':>7} {'n':>5} {'|dIV|bps med/p90':>18} {'|dDelta| med/p90':>18} "
          f"{'|dGam|/G med/p90':>18} {'|dTheta| med/p90':>18} {'dIV bps med':>12} "
          f"{'offset h med [p10..p90] (n)':>30}")
    nan = float("nan")
    for row in r["rows"]:
        def pair(k: str) -> str:
            return f"{row.get(k + '_med', nan):.4g}/{row.get(k + '_p90', nan):.4g}"
        if "n_offset" in row:
            off = (f"{row['offset_h_med']:+.1f} [{row['offset_h_p10']:+.1f}..{row['offset_h_p90']:+.1f}]"
                   f" ({row['n_offset']})")
        else:
            off = "-"
        print(f"{row['moneyness']:>10} {row['dte']:>7} {row['n']:>5} {pair('iv_bps'):>18} "
              f"{pair('delta'):>18} {pair('gamma_rel'):>18} {pair('theta'):>18} "
              f"{row.get('iv_signed_bps_med', nan):>+12.4g} {off:>30}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--interval", type=float, default=300.0, help="seconds between diff reports")
    ap.add_argument("--max-skew-s", type=float, default=60.0, help="max receipt-time gap for a pair")
    ap.add_argument("--jsonl", type=Path, default=None, help="also append every report here")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-5s %(name)s - %(message)s")

    bus = ServiceLocator.bus()
    sub = ResilientSub("GreeksShadow", bus.xpub, [VENDOR, LOCAL, GREEKS_HB], MDS_HB, recv_timeout_s=1.0)
    shadow = Shadow(args.max_skew_s)
    next_report = time.monotonic() + args.interval
    sink = args.jsonl.open("a", encoding="utf-8") if args.jsonl else None
    log.info("listening on %s for %s and %s", bus.xpub, VENDOR, LOCAL)
    try:
        while True:
            topic, payload = sub.recv()
            now = time.monotonic()
            if topic:
                try:
                    shadow.on(topic, payload, now)
                except (ValueError, KeyError, TypeError) as e:
                    log.warning("bad payload on %s: %s", topic, e)
            records = shadow.coverage(now)
            if now >= next_report:
                r = shadow.report()
                _print_report(r)
                records.append(r)
                next_report = now + args.interval
            for rec in records:
                if rec["kind"] == "coverage":
                    print(f"coverage +{rec['after_s']}s after restart: {rec['no_local']} of "
                          f"{rec['vendor_live']} vendor-live contracts have no local greeks {rec['sample']}")
                if sink:
                    sink.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(), **rec}) + "\n")
                    sink.flush()
    except KeyboardInterrupt:
        pass
    finally:
        sub.close()
        if sink:
            sink.close()


if __name__ == "__main__":
    main()
