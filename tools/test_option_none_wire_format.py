"""
Test upickle's Option[None] serialization behavior in CalcServer responses.

Calls list_metrics via the control router and inspects whether parameters
with None-valued optional fields (min, max) have those fields:
  - absent from the JSON (matches existing wire format)
  - present as null (new behavior introduced by upickle migration)

The script uses discovery to find CalcServer, sends a DEALER->ROUTER RPC
request, and reports the finding.

Usage:
    PYTHONPATH=. python tools/test_option_none_wire_format.py

Exit status:
    0 — optional fields are omitted when None (matches existing format)
    1 — optional fields are present as null (wire format change)
    2 — no parameters with None-valued optional fields found in response
    3 — discovery or RPC failure
"""

import json
import sys

import zmq

from control_client import ControlSocket, ControlTimeout

from discovery.service_locator import ServiceLocator


def main() -> int:
    # Discover CalcServer's control router endpoint
    try:
        endpoint = ServiceLocator.wait_for_service(
            ServiceLocator.CALC,
            timeout_sec=10,
        )
    except Exception as e:
        print(f"Discovery failed: {e}", file=sys.stderr)
        return 3

    router_endpoint = endpoint.router
    if not router_endpoint:
        print(f"CalcServer discovery returned no router endpoint: {endpoint}", file=sys.stderr)
        return 3

    print(f"CalcServer control router: {router_endpoint}")

    # Send list_metrics request
    sock = ControlSocket(router_endpoint, timeout_ms=5000, ctx=zmq.Context.instance())
    try:
        response = sock.request({"op": "list_metrics"})
    except ControlTimeout:
        print("RPC timeout waiting for list_metrics reply", file=sys.stderr)
        return 3
    finally:
        sock.close()

    payload = json.dumps(response)

    print("\n--- Raw response (first 2000 chars) ---")
    print(payload[:2000])
    print("--- End raw response ---\n")

    try:
        response = json.loads(payload)
    except json.JSONDecodeError as e:
        print(f"Response is not valid JSON: {e}", file=sys.stderr)
        return 3

    if not response.get("ok"):
        print(f"RPC returned error: {response.get('error')}", file=sys.stderr)
        return 3

    metrics = response.get("data", {}).get("metrics", [])
    if not metrics:
        print("No metrics in response; cannot verify Option behavior", file=sys.stderr)
        return 2

    # Inspect parameters for Option[None] behavior
    null_count = 0
    absent_count = 0
    first_absent_sample = None
    first_null_sample = None

    for metric in metrics:
        for param in metric.get("parameters", []):
            has_min_key = "min" in param
            has_max_key = "max" in param
            min_is_null = param.get("min") is None and has_min_key
            max_is_null = param.get("max") is None and has_max_key

            if min_is_null or max_is_null:
                null_count += 1
                if first_null_sample is None:
                    first_null_sample = (metric["name"], param)
            elif not has_min_key or not has_max_key:
                # At least one optional field absent
                absent_count += 1
                if first_absent_sample is None:
                    first_absent_sample = (metric["name"], param)

    print(f"Parameters with null-valued optional fields: {null_count}")
    print(f"Parameters with absent optional fields:      {absent_count}")

    if null_count > 0 and absent_count == 0:
        print("\nWIRE FORMAT CHANGE DETECTED: upickle is emitting null for None")
        print(f"Sample ({first_null_sample[0]}): {json.dumps(first_null_sample[1], indent=2)}")
        print("\nFix: configure upickle to omit None fields rather than emit null.")
        return 1

    if absent_count > 0 and null_count == 0:
        print("\nOK: optional fields are omitted when None (matches existing wire format)")
        if first_absent_sample:
            print(f"Sample ({first_absent_sample[0]}): {json.dumps(first_absent_sample[1], indent=2)}")
        return 0

    if null_count > 0 and absent_count > 0:
        print("\nMIXED: some optional fields emit null, others are absent")
        print("This suggests inconsistent codec behavior — worth investigating.")
        if first_null_sample:
            print(f"\nNull sample ({first_null_sample[0]}): {json.dumps(first_null_sample[1], indent=2)}")
        if first_absent_sample:
            print(f"\nAbsent sample ({first_absent_sample[0]}): {json.dumps(first_absent_sample[1], indent=2)}")
        return 1

    # null_count == 0 and absent_count == 0 — every parameter had both min and max set
    print("\nNo parameters found with None-valued optional fields.")
    print("Cannot verify Option[None] behavior from list_metrics alone.")
    print("All parameters in the registry have both min and max defined.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
