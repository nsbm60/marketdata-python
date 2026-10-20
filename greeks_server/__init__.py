"""Live greeks server (Scala/MarketData docs/specs/greeks-server-spec.md).

Computes IV and greeks for the contracts the platform is streaming and publishes them on the bus;
answers compute_greeks for contracts that aren't streamed. Pricing comes from the certified
``greeks`` solver; this package adds none of its own. It lives outside ``greeks/`` because it
uses threads (the service library), which ``greeks/`` forbids.

Modules
  calendar  session hours (port of the Scala USMarketCalendar)
  wire      decode MDS tick payloads
  price     price selection (§4)
  inputs    SOFR + dividends per session
  compute   price one contract; the iv_for seam
  message   OptionGreeksMessage (the payload)
  book      per-session ticks and the dirty set
  loop      the compute thread
  handler   compute_greeks
"""
