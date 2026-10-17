#!/usr/bin/env python3
"""
alpaca_feed_probe.py

Open one or more Alpaca market-data websocket feeds, authenticate, subscribe, and
report what happens. Written to answer three questions about the 24/5 `boats`
feed, in the order they matter:

  1. Are we entitled to `boats` at all?           (answerable any time of day)
  2. Can `boats` coexist with a `sip` connection? (the design decision)
  3. Does `boats` actually deliver overnight?     (needs a run after 8pm ET)

The feed is part of the URL, not a parameter, so each feed is a separate
connection — which is why (2) is a real question. Alpaca limits connections and
returns error 406 when the limit is hit.

IMPORTANT: silence is not failure. Outside 8pm-4am ET `boats` carries nothing,
and during that window `sip` carries nothing. A feed that authenticates and
acknowledges the subscription has passed the entitlement test even if no message
ever arrives. This script reports those two things separately for that reason.

Usage:
    export ALPACA_API_KEY=...  ALPACA_API_SECRET=...

    # With MDS running — does boats coexist with the sip connection MDS holds?
    python tools/alpaca_feed_probe.py

    # With MDS stopped — can this process hold both at once?
    python tools/alpaca_feed_probe.py --feeds sip,boats

    # After 8pm ET — does data actually flow?
    python tools/alpaca_feed_probe.py --seconds 120
"""

import argparse
import asyncio
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone

try:
    import websockets
except ImportError:
    print("websockets not installed. Try: pip install websockets", file=sys.stderr)
    sys.exit(1)

BASE = "wss://stream.data.alpaca.markets"

# Feed -> URL path. `boats` is the 24/5 overnight feed on Algo Trader Plus;
# `overnight` is the free-plan equivalent (indicative quotes, delayed trades).
# NOTE the version differs: the 24/5 feeds are v1beta1, the regular ones v2.
# Same split as the OPRA options stream (/v1beta1/opra) in SystemConfig.
FEED_PATHS = {
    "sip":         "/v2/sip",
    "iex":         "/v2/iex",
    "delayed_sip": "/v2/delayed_sip",
    "boats":       "/v1beta1/boats",
    "overnight":   "/v1beta1/overnight",
}

# Alpaca stream error codes worth naming rather than printing raw.
ERROR_MEANINGS = {
    400: "invalid syntax",
    401: "not authenticated",
    402: "auth failed — check key/secret",
    403: "already authenticated",
    404: "auth timed out",
    405: "symbol limit exceeded",
    406: "CONNECTION LIMIT EXCEEDED — this feed cannot be held alongside the others",
    407: "slow client",
    408: "v2 data not enabled on this account",
    409: "INSUFFICIENT SUBSCRIPTION — the plan does not include this feed",
    410: "invalid subscribe action",
}


def stamp() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%H:%M:%S")


def format_trade(m: dict) -> str:
    """One trade, laid out so a run of them is readable in a terminal."""
    conds = ",".join(m.get("c") or []) or "-"
    return (f"TRADE {m.get('S',''):6s} {m.get('p', 0):>10,.4f} x {m.get('s', 0):<8,} "
            f"exch={m.get('x','?'):2s} cond={conds:8s} id={m.get('i','')}")


def format_quote(m: dict) -> str:
    return (f"QUOTE {m.get('S',''):6s} {m.get('bp', 0):>10,.4f} x {m.get('bs', 0):<6,} / "
            f"{m.get('ap', 0):>10,.4f} x {m.get('as', 0):<6,}")


def track_timing(result: "FeedResult", m: dict) -> str:
    """Record how stale a message is, and whether it went backwards.

    Alpaca stamps each trade/quote with its own market time in `t` (RFC3339).
    Comparing that to the local clock separates a live print from a cached
    re-delivery, and comparing it to the newest timestamp already seen for that
    symbol catches a stale value arriving after a fresh one.
    """
    ts = m.get("t")
    sym = m.get("S")
    if not ts or not sym:
        return ""
    try:
        when = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return f"   [unparseable timestamp {ts!r}]"

    lag_ms = (datetime.now(timezone.utc) - when).total_seconds() * 1000.0
    result.lags_ms.append(lag_ms)

    note = f"   [age {lag_ms:,.0f}ms]"
    prev = result.latest_ts.get(sym)
    if prev is not None and when < prev:
        result.out_of_order += 1
        note += f"  <-- OLDER than a message already seen for {sym}"
    else:
        result.latest_ts[sym] = when
    return note


class FeedResult:
    def __init__(self, feed: str):
        self.feed = feed
        self.connected = False
        self.authenticated = False
        self.subscribed = None      # the subscription ack payload
        self.error = None           # (code, msg)
        self.counts = Counter()     # message type -> count
        self.first_sample = {}      # message type -> first message seen
        # Message age: how far behind the local clock each print's own market
        # timestamp is. A real tape stays close to zero and never goes backwards.
        # Re-delivered cached values — the thing that made IB unusable — show up
        # here as large ages or as timestamps older than one already seen.
        self.lags_ms = []
        self.out_of_order = 0
        self.latest_ts = {}         # symbol -> newest market timestamp seen

    @property
    def entitled(self) -> bool:
        return self.authenticated and self.subscribed is not None


async def probe(result: FeedResult, symbols: list[str], seconds: int, verbose: bool) -> FeedResult:
    """Run one feed. `seconds <= 0` means run until cancelled.

    The result object is passed in rather than created here so that the caller
    still has the counts and samples if this task is cancelled by Ctrl-C.
    """
    feed = result.feed
    url = BASE + FEED_PATHS[feed]

    try:
        async with websockets.connect(url, ping_interval=20, max_size=None) as ws:
            print(f"[{stamp()}] {feed:9s} socket open -> {url}")

            # 1. Server greets with [{"T":"success","msg":"connected"}]
            greeting = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
            for m in greeting:
                if m.get("T") == "success" and m.get("msg") == "connected":
                    result.connected = True
                elif m.get("T") == "error":
                    result.error = (m.get("code"), m.get("msg"))

            if result.error:
                return result

            # 2. Authenticate.
            await ws.send(json.dumps({
                "action": "auth",
                "key": os.environ["ALPACA_API_KEY"],
                "secret": os.environ["ALPACA_API_SECRET"],
            }))
            auth = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
            for m in auth:
                if m.get("T") == "success" and m.get("msg") == "authenticated":
                    result.authenticated = True
                    print(f"[{stamp()}] {feed:9s} authenticated")
                elif m.get("T") == "error":
                    result.error = (m.get("code"), m.get("msg"))
                    code, msg = result.error
                    print(f"[{stamp()}] {feed:9s} AUTH ERROR {code}: {msg} "
                          f"-- {ERROR_MEANINGS.get(code, 'unknown code')}")

            if not result.authenticated:
                return result

            # 3. Subscribe. Trades and quotes only; bars would just add noise.
            await ws.send(json.dumps({
                "action": "subscribe",
                "trades": symbols,
                "quotes": symbols,
            }))

            # 4. Listen. The subscription ack arrives on this stream too.
            loop = asyncio.get_event_loop()
            deadline = None if seconds <= 0 else loop.time() + seconds
            if deadline is None:
                print(f"[{stamp()}] {feed:9s} listening — Ctrl-C to stop")

            while True:
                if deadline is not None:
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        break
                else:
                    remaining = None
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
                except asyncio.TimeoutError:
                    break

                for m in json.loads(raw):
                    t = m.get("T")
                    result.counts[t] += 1
                    if t == "subscription":
                        result.subscribed = {k: v for k, v in m.items() if k != "T"}
                        print(f"[{stamp()}] {feed:9s} subscription ack: {result.subscribed}")
                    elif t == "error":
                        result.error = (m.get("code"), m.get("msg"))
                        code, msg = result.error
                        print(f"[{stamp()}] {feed:9s} ERROR {code}: {msg} "
                              f"-- {ERROR_MEANINGS.get(code, 'unknown code')}")
                    else:
                        note = track_timing(result, m)
                        result.first_sample.setdefault(t, m)
                        if t == "t":
                            print(f"[{stamp()}] {feed:9s} {format_trade(m)}{note}")
                        elif t == "q":
                            if verbose:
                                print(f"[{stamp()}] {feed:9s} {format_quote(m)}{note}")
                        elif verbose or t not in result.first_sample:
                            print(f"[{stamp()}] {feed:9s} {t}: {m}{note}")

    except Exception as e:
        result.error = (None, f"{type(e).__name__}: {e}")
        print(f"[{stamp()}] {feed:9s} connection failed: {result.error[1]}")

    return result


async def main_async(args) -> int:
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    feeds = [f.strip().lower() for f in args.feeds.split(",") if f.strip()]

    for f in feeds:
        if f not in FEED_PATHS:
            print(f"Unknown feed {f!r}. Known: {', '.join(FEED_PATHS)}", file=sys.stderr)
            return 2

    print(f"Probing {', '.join(feeds)} with {symbols} for {args.seconds}s")
    print("Feeds are opened concurrently — if the account allows only one connection")
    print("overall, the second to authenticate should come back with error 406.\n")

    results = [FeedResult(f) for f in feeds]
    try:
        await asyncio.gather(*(probe(r, symbols, args.seconds, args.verbose) for r in results))
    except asyncio.CancelledError:
        pass

    print_summary(results)
    return 0


def print_summary(results: list[FeedResult]) -> int:
    print("\n" + "=" * 70)
    for r in results:
        print(f"\n{r.feed}")
        print(f"  socket open       {r.connected}")
        print(f"  authenticated     {r.authenticated}")
        print(f"  subscription ack  {r.subscribed if r.subscribed else 'none'}")
        if r.error:
            code, msg = r.error
            meaning = ERROR_MEANINGS.get(code, "")
            print(f"  error             {code}: {msg}" + (f"  [{meaning}]" if meaning else ""))
        data_msgs = sum(v for k, v in r.counts.items() if k in ("t", "q", "b"))
        print(f"  data messages     {data_msgs}  ({dict(r.counts) or 'none'})")
        print(f"  -> entitled       {'YES' if r.entitled else 'NO'}")
        flowing = "YES" if data_msgs else "no (expected outside this feed's window)"
        print(f"  -> data flowing   {flowing}")
        if r.lags_ms:
            r.lags_ms.sort()
            median = r.lags_ms[len(r.lags_ms) // 2]
            print(f"  message age       median {median:,.0f}ms  "
                  f"min {r.lags_ms[0]:,.0f}ms  max {r.lags_ms[-1]:,.0f}ms")
            print(f"  out of order      {r.out_of_order}"
                  + ("  <-- stale values arriving after fresh ones" if r.out_of_order else ""))

    print("\n" + "=" * 70)
    entitled = [r.feed for r in results if r.entitled]
    refused = [(r.feed, r.error) for r in results if not r.entitled]
    if len(entitled) == len(results):
        print(f"All feeds entitled and subscribed: {', '.join(entitled)}")
        if len(results) > 1:
            print("They can be held simultaneously — no connection-limit conflict.")
    else:
        for feed, err in refused:
            code = err[0] if err else None
            if code == 406:
                print(f"{feed}: refused with 406 — cannot coexist with the other connection(s).")
            elif code == 409:
                print(f"{feed}: refused with 409 — the plan does not include this feed.")
            else:
                print(f"{feed}: did not reach a subscribed state ({err}).")
    print("\nReminder: no data outside a feed's own window is expected, not a failure.")
    print("boats carries data 8pm-4am ET; sip carries data 4am-8pm ET.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Probe Alpaca market-data websocket feeds")
    ap.add_argument("--feeds", default="boats",
                    help="comma-separated: boats, sip, iex, overnight (default: boats)")
    ap.add_argument("--symbols", default="MU,NVDA")
    ap.add_argument("--seconds", type=int, default=0,
                    help="how long to listen; 0 (default) runs until Ctrl-C")
    ap.add_argument("--verbose", action="store_true",
                    help="also print quotes (trades are always printed)")
    args = ap.parse_args()

    if not os.environ.get("ALPACA_API_KEY") or not os.environ.get("ALPACA_API_SECRET"):
        print("Set ALPACA_API_KEY and ALPACA_API_SECRET", file=sys.stderr)
        return 2

    try:
        return asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\n\nstopped.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
