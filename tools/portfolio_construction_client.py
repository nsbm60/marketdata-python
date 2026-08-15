#!/usr/bin/env python3
"""
Portfolio construction report subscriber and validator.

Subscribes to report.portfolio_construction.<broker> on CalcServer (port 6020),
pretty-prints each message, and optionally validates schema invariants or
cross-checks against the positions report.

Both portfolio construction and positions reports are always-on (bounded) —
no start/stop RPCs needed.

Examples:
  # Stream reports, pretty-printed (default broker: ib)
  python portfolio_construction_client.py

  # Validate schema invariants, exit after 1 message
  python portfolio_construction_client.py --validate --count 1

  # Cross-check against positions report
  python portfolio_construction_client.py --cross-check --count 1

  # Write 10 messages as JSON lines to a file
  python portfolio_construction_client.py --count 10 --output report.jsonl

  # Force an immediate publish before subscribing
  python portfolio_construction_client.py --publish-now
"""

import argparse
import json
import signal
import sys
import time

import zmq

CALC_PUB_PORT = 6020
CALC_CONTROL_PORT = 6021
TOLERANCE = 0.01  # absolute tolerance for floating-point invariant checks
PCT_TOLERANCE = 0.5  # percentage points tolerance for pct sums
CROSS_CHECK_TOLERANCE = 0.05  # 5% relative tolerance for cross-check


def rpc(ctx, host, op, payload):
    """Send an RPC to CalcServer's control router and return the parsed response."""
    dealer = ctx.socket(zmq.DEALER)
    dealer.setsockopt(zmq.RCVTIMEO, 5000)
    dealer.connect(f"tcp://{host}:{CALC_CONTROL_PORT}")
    msg = json.dumps({"op": op, **payload})
    dealer.send_multipart([b"", msg.encode()])
    try:
        parts = dealer.recv_multipart()
        return json.loads(parts[-1])
    except zmq.Again:
        return {"ok": False, "error": "RPC timeout"}
    finally:
        dealer.close()


def validate(r):
    """Run schema invariants on a report. Returns list of failure strings."""
    failures = []

    # Required top-level keys (capital may be absent when account metrics unavailable)
    required = {"account", "asOf", "sessionState", "status", "aggregateGreeks",
                "perUnderlying", "thetaProfile", "concentration", "dataQuality"}
    missing = required - set(r.keys())
    if missing:
        failures.append(f"missing top-level keys: {missing}")

    if r.get("status") == "error":
        return failures  # error reports don't have the rest

    ag = r.get("aggregateGreeks", {})
    total_theta = ag.get("totalThetaDaily", 0)

    # Σ perUnderlying.thetaDaily == totalThetaDaily
    pu = r.get("perUnderlying", [])
    pu_sum = sum(u["thetaDaily"] for u in pu)
    if abs(pu_sum - total_theta) > TOLERANCE:
        failures.append(f"perUnderlying theta sum {pu_sum:.4f} != totalThetaDaily {total_theta:.4f}")

    # Σ thetaProfile.buckets.thetaDaily == totalThetaDaily
    buckets = r.get("thetaProfile", {}).get("buckets", [])
    bucket_sum = sum(b["thetaDaily"] for b in buckets)
    if abs(bucket_sum - total_theta) > TOLERANCE:
        failures.append(f"bucket theta sum {bucket_sum:.4f} != totalThetaDaily {total_theta:.4f}")

    # Σ concentration.byExpiryWeek.thetaDaily == totalThetaDaily
    weeks = r.get("concentration", {}).get("byExpiryWeek", [])
    week_sum = sum(w["thetaDaily"] for w in weeks)
    if abs(week_sum - total_theta) > TOLERANCE:
        failures.append(f"week theta sum {week_sum:.4f} != totalThetaDaily {total_theta:.4f}")

    # Σ pctOfAbsDollarDelta == 100
    if pu:
        pct_sum = sum(u["pctOfAbsDollarDelta"] for u in pu)
        if abs(pct_sum - 100) > PCT_TOLERANCE:
            failures.append(f"pctOfAbsDollarDelta sum {pct_sum:.2f} != 100")

    # Σ thetaProfile.buckets.pct == 100 (when there are options)
    if buckets and total_theta != 0:
        bpct_sum = sum(b["pct"] for b in buckets)
        if abs(bpct_sum - 100) > PCT_TOLERANCE:
            failures.append(f"bucket pct sum {bpct_sum:.2f} != 100")

    # Σ concentration.byExpiryWeek.pct == 100
    if weeks and total_theta != 0:
        wpct_sum = sum(w["pct"] for w in weeks)
        if abs(wpct_sum - 100) > PCT_TOLERANCE:
            failures.append(f"week pct sum {wpct_sum:.2f} != 100")

    # capital.buyingPowerUsed == netLiq - buyingPower
    cap = r.get("capital")
    if cap:
        expected_bpu = cap["netLiq"] - cap["buyingPower"]
        if abs(cap["buyingPowerUsed"] - expected_bpu) > TOLERANCE:
            failures.append(f"buyingPowerUsed {cap['buyingPowerUsed']:.2f} != netLiq - buyingPower {expected_bpu:.2f}")

    return failures


def cross_check(ctx, host, broker, pc_report):
    """Subscribe to the always-on positions report, capture one message, reconcile per-underlying theta."""
    topic = f"report.positions.{broker}"
    sub = ctx.socket(zmq.SUB)
    sub.connect(f"tcp://{host}:{CALC_PUB_PORT}")
    sub.setsockopt_string(zmq.SUBSCRIBE, topic)

    poller = zmq.Poller()
    poller.register(sub, zmq.POLLIN)
    pos_report = None
    start = time.time()
    while time.time() - start < 10:
        socks = dict(poller.poll(timeout=1000))
        if sub in socks:
            sub.recv_string()  # topic
            pos_report = json.loads(sub.recv_string())
            break
    sub.close()

    if pos_report is None:
        print("  Timeout waiting for positions report", file=sys.stderr)
        return False

    # Aggregate per-underlying theta from positions report (options with Greeks only)
    pos_theta = {}
    for p in pos_report.get("positions", []):
        if p.get("secType") == "OPT" and "thetaDaily" in p:
            und = p.get("underlying", p.get("symbol", "?"))
            pos_theta[und] = pos_theta.get(und, 0) + p["thetaDaily"]

    # Per-underlying theta from portfolio construction report
    pc_theta = {u["underlying"]: u["thetaDaily"] for u in pc_report.get("perUnderlying", [])}

    # Reconciliation table
    all_underlyings = sorted(set(pos_theta.keys()) | set(pc_theta.keys()))
    print(f"\n  {'Underlying':>10s}  {'Pos Theta':>10s}  {'PC Theta':>10s}  {'Diff':>10s}  {'Pct':>8s}  Status")
    print(f"  {'─' * 10}  {'─' * 10}  {'─' * 10}  {'─' * 10}  {'─' * 8}  {'─' * 6}")

    ok = True
    for und in all_underlyings:
        pt = pos_theta.get(und, 0)
        pct = pc_theta.get(und, 0)
        diff = pct - pt
        rel = abs(diff / pt) if pt != 0 else (0 if pct == 0 else float("inf"))
        status = "OK" if rel <= CROSS_CHECK_TOLERANCE else "WARN"
        if rel > CROSS_CHECK_TOLERANCE:
            ok = False
        print(f"  {und:>10s}  {pt:>10.2f}  {pct:>10.2f}  {diff:>+10.2f}  {rel:>7.1%}  {status}")

    return ok


def main():
    parser = argparse.ArgumentParser(description="Portfolio construction report client.")
    parser.add_argument("--broker", default="ib", help="Broker wire name (default: ib)")
    parser.add_argument("--host", default="localhost", help="CalcServer host (default: localhost)")
    parser.add_argument("--publish-now", action="store_true", help="Force immediate publish via RPC before subscribing")
    parser.add_argument("--validate", action="store_true", help="Run schema invariants on each message")
    parser.add_argument("--cross-check", action="store_true", help="Cross-check against positions report")
    parser.add_argument("--count", type=int, default=0, help="Exit after N messages (0 = unlimited)")
    parser.add_argument("--output", help="Write messages as JSON lines to file")
    args = parser.parse_args()

    ctx = zmq.Context()

    if args.publish_now:
        resp = rpc(ctx, args.host, "publish_portfolio_construction_report_now", {"broker": args.broker})
        status = resp.get("data", {}).get("status", resp.get("error", "unknown"))
        print(f"RPC publish_now: {status}", file=sys.stderr)

    topic = f"report.portfolio_construction.{args.broker}"
    sub = ctx.socket(zmq.SUB)
    sub.connect(f"tcp://{args.host}:{CALC_PUB_PORT}")
    sub.setsockopt_string(zmq.SUBSCRIBE, topic)

    outfile = open(args.output, "w") if args.output else None
    msg_count = 0
    exit_code = 0

    def cleanup(sig=None, frame=None):
        sub.close()
        ctx.term()
        if outfile:
            outfile.close()
        sys.exit(exit_code)

    signal.signal(signal.SIGINT, cleanup)
    print(f"Subscribed to {topic}", file=sys.stderr)

    while True:
        sub.recv_string()  # topic
        payload = sub.recv_string()
        msg_count += 1
        r = json.loads(payload)

        if outfile:
            outfile.write(payload + "\n")
            outfile.flush()
        else:
            print(json.dumps(r, indent=2))

        if args.validate:
            failures = validate(r)
            if failures:
                exit_code = 1
                for f in failures:
                    print(f"  FAIL: {f}", file=sys.stderr)
            else:
                print(f"  PASS: all invariants hold", file=sys.stderr)

        if args.cross_check:
            ok = cross_check(ctx, args.host, args.broker, r)
            if not ok:
                exit_code = 1

        if args.count and msg_count >= args.count:
            break

    cleanup()


if __name__ == "__main__":
    main()
