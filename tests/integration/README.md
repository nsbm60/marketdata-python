# Integration tests

Live tests that run against **running services** (MDS, CalcServer, discovery) — distinct from the
ad-hoc probes and one-off scripts in `tools/`. They discover services via ZMQ
(`discovery.service_locator`) rather than hard-coded endpoints.

**Prerequisites:** MDS + discovery up and reachable from where you run the test.

Run a test standalone:

```
python tests/integration/test_option_subscriptions.py
```

or under pytest (`conftest.py` puts the repo root on the path):

```
pytest tests/integration
```

## test_option_subscriptions.py

Queries MDS's `option_subscriptions` control op — what MDS is currently providing option data for,
the set that drives the option poller **and** the option stream — and reports subscribed contracts
grouped by underlying with their TTLs. Verifies subscriptions are actually tracked, not just that
streaming happens.

- Open an options chain in the UI, then run: that underlying's contracts should appear, with TTLs
  near the lease duration (~10 min) and `missed_inquiries = 0`.
- Close the chain / disconnect: after the grace period the contracts lapse and the count drops.

## Candidates to migrate from `tools/`

These are really integration tests and could move here later (left in place for now):

- `test_service_infra.py` — service discovery + control echo
- `test_option_stream_vs_rest.py`
- `test_option_none_wire_format.py`
- `test_alpaca_opra_direct.py`
