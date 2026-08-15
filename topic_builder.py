"""TopicBuilder — the single source of truth for ZMQ topic strings on the Python side.

Mirrors the Scala `com.nsbm.marketdata.topic.TopicBuilder`. Build topics through these methods,
never with inline string literals — a topic-format change is then one edit here, not a hunt across
tools, and casing conventions (which have bitten us, e.g. `service.marketdata`) live in exactly one
place.

Wire conventions (matching Scala):
  - Discovery topics lowercase the service name:      service.<name lower>   (TopicBuilder.forService)
  - Service heartbeat uses the name verbatim:         <name>.heartbeat       (forServiceHeartbeat)
  - Equity/option symbols are uppercase on the wire;  report underlyings are lowercase.
"""
from __future__ import annotations


class TopicBuilder:
    # ── domains / categories / parts (the vocabulary; no bare literals elsewhere) ──
    DOMAIN_MARKET = "md"
    DOMAIN_SERVICE = "service"   # discovery bus
    DOMAIN_IB = "ib"
    DOMAIN_REPORT = "report"     # pre-computed reports from CalcServer

    CAT_EQUITY = "equity"
    CAT_OPTION = "option"

    PART_TRADE = "trade"
    PART_QUOTE = "quote"
    PART_GREEKS = "greeks"
    PART_BAR = "bar"
    PART_INDICATOR = "indicator"
    PART_EMA = "ema"
    PART_ATR = "atr"
    PART_PIVOT = "pivot"
    PART_OPTIONS = "options"
    PART_PORTFOLIO = "portfolio"
    PART_POSITIONS = "positions"
    PART_WATCHLIST = "watchlist"

    DOT = "."

    # ── discovery bus ────────────────────────────────────────────────────────────
    @staticmethod
    def for_service(name: str) -> str:
        """Discovery topic for a service. Lowercases the name (Scala `forService` does the same),
        so callers may pass the canonical `marketData` and still match `service.marketdata`."""
        return f"{TopicBuilder.DOMAIN_SERVICE}{TopicBuilder.DOT}{name.lower()}"

    @staticmethod
    def service_prefix() -> str:
        """Prefix for the discovery bus, e.g. to strip from a received topic."""
        return f"{TopicBuilder.DOMAIN_SERVICE}{TopicBuilder.DOT}"

    @staticmethod
    def for_service_heartbeat(name: str) -> str:
        """`<name>.heartbeat` — verbatim, matching Scala `forServiceHeartbeat` (note: NOT lowercased
        there, unlike the discovery topic)."""
        return f"{name}{TopicBuilder.DOT}heartbeat"

    # ── equity market data (symbols uppercase) ─────────────────────────────────────
    @staticmethod
    def for_equity_trade(symbol: str) -> str:
        return f"{TopicBuilder.DOMAIN_MARKET}.{TopicBuilder.CAT_EQUITY}.{TopicBuilder.PART_TRADE}.{symbol.upper()}"

    @staticmethod
    def for_equity_quote(symbol: str) -> str:
        return f"{TopicBuilder.DOMAIN_MARKET}.{TopicBuilder.CAT_EQUITY}.{TopicBuilder.PART_QUOTE}.{symbol.upper()}"

    @staticmethod
    def for_equity_bar(timeframe: str, symbol: str) -> str:
        return f"{TopicBuilder.DOMAIN_MARKET}.{TopicBuilder.CAT_EQUITY}.{TopicBuilder.PART_BAR}.{timeframe}.{symbol.upper()}"

    @staticmethod
    def for_equity_indicator_ema(timeframe: str, symbol: str) -> str:
        return f"{TopicBuilder.DOMAIN_MARKET}.{TopicBuilder.CAT_EQUITY}.{TopicBuilder.PART_INDICATOR}.{TopicBuilder.PART_EMA}.{timeframe}.{symbol.upper()}"

    @staticmethod
    def for_equity_indicator_atr(timeframe: str, symbol: str) -> str:
        return f"{TopicBuilder.DOMAIN_MARKET}.{TopicBuilder.CAT_EQUITY}.{TopicBuilder.PART_INDICATOR}.{TopicBuilder.PART_ATR}.{timeframe}.{symbol.upper()}"

    @staticmethod
    def for_equity_pivot(timeframe: str, symbol: str) -> str:
        return f"{TopicBuilder.DOMAIN_MARKET}.{TopicBuilder.CAT_EQUITY}.{TopicBuilder.PART_PIVOT}.{timeframe}.{symbol.upper()}"

    @staticmethod
    def prefix_equity_quotes() -> str:
        return f"{TopicBuilder.DOMAIN_MARKET}.{TopicBuilder.CAT_EQUITY}.{TopicBuilder.PART_QUOTE}."

    @staticmethod
    def prefix_equity_trades() -> str:
        return f"{TopicBuilder.DOMAIN_MARKET}.{TopicBuilder.CAT_EQUITY}.{TopicBuilder.PART_TRADE}."

    # ── option market data — SUB prefixes (Python consumes by prefix; underlying upper) ──
    @staticmethod
    def prefix_option_quotes() -> str:
        return f"{TopicBuilder.DOMAIN_MARKET}.{TopicBuilder.CAT_OPTION}.{TopicBuilder.PART_QUOTE}."

    @staticmethod
    def prefix_option_trades() -> str:
        return f"{TopicBuilder.DOMAIN_MARKET}.{TopicBuilder.CAT_OPTION}.{TopicBuilder.PART_TRADE}."

    @staticmethod
    def prefix_option_greeks() -> str:
        return f"{TopicBuilder.DOMAIN_MARKET}.{TopicBuilder.CAT_OPTION}.{TopicBuilder.PART_GREEKS}."

    @staticmethod
    def prefix_option_trades_for(underlying: str) -> str:
        return f"{TopicBuilder.prefix_option_trades()}{underlying.upper()}."

    @staticmethod
    def prefix_option_quotes_for(underlying: str) -> str:
        return f"{TopicBuilder.prefix_option_quotes()}{underlying.upper()}."

    # ── CalcServer reports (underlying lowercase, matching Scala forOptionsReport) ──
    @staticmethod
    def for_options_report(underlying: str, expiry: str) -> str:
        return f"{TopicBuilder.DOMAIN_REPORT}.{TopicBuilder.PART_OPTIONS}.{underlying.lower()}.{expiry}"

    @staticmethod
    def prefix_options_reports() -> str:
        return f"{TopicBuilder.DOMAIN_REPORT}.{TopicBuilder.PART_OPTIONS}."

    @staticmethod
    def prefix_portfolio_options_reports() -> str:
        return f"{TopicBuilder.DOMAIN_REPORT}.{TopicBuilder.PART_PORTFOLIO}.{TopicBuilder.PART_OPTIONS}."

    @staticmethod
    def for_positions_report(broker: str) -> str:
        return f"{TopicBuilder.DOMAIN_REPORT}.{TopicBuilder.PART_POSITIONS}.{broker}"

    @staticmethod
    def report_prefix() -> str:
        return f"{TopicBuilder.DOMAIN_REPORT}{TopicBuilder.DOT}"
