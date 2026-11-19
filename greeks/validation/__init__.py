"""Read-only validation probes that compare our greeks/IV with a vendor's.

Nothing here writes to ClickHouse. `massive_analytics` compares our option_trade greeks against
Massive's Historical Greeks & IV endpoint (GET /options/v1/analytics).
"""
