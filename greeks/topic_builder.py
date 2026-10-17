"""TopicBuilder — the single source of truth for ZMQ topic strings on the Python side.

Mirrors the Scala `com.nsbm.marketdata.topic.TopicBuilder`. Build topics through these methods,
never with inline string literals — a topic-format change is then one edit here, not a hunt across
tools, and casing conventions (which have bitten us, e.g. `service.marketdata`) live in exactly one
place.

Wire conventions (matching Scala):
  - Discovery topics lowercase the service name:      service.<name lower>   (TopicBuilder.forService)
  - Service heartbeat uses the name verbatim:         <name>.heartbeat       (forServiceHeartbeat)
  - Equity/option symbols are uppercase on the wire;  report underlyings are lowercase.
  - Option topics are hierarchical:                   md.option.<part>.UNDERLYING.EXPIRY.SIDE.STRIKE
        e.g. md.option.greeks.NVDA.2025-01-03.C.177_50   (Scala optionSuffix / formatStrike)

The cross-language golden test (tests/greeks/test_topic_golden.py, fixture
tests/greeks/topic_golden.json) pins the option producers to the exact strings the
Scala TopicBuilder emits. Do not change a format here without regenerating that fixture
from the Scala builder.
"""
from __future__ import annotations

from datetime import date
from decimal import ROUND_HALF_UP, Decimal


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
        return f"{TopicBuilder.DOMAIN_MARKET}.{TopicBuilder.CAT_EQUITY}.{TopicBuilder.PART_TRADE}.{TopicBuilder._normalize_symbol(symbol)}"

    @staticmethod
    def for_equity_quote(symbol: str) -> str:
        return f"{TopicBuilder.DOMAIN_MARKET}.{TopicBuilder.CAT_EQUITY}.{TopicBuilder.PART_QUOTE}.{TopicBuilder._normalize_symbol(symbol)}"

    @staticmethod
    def for_equity_bar(timeframe: str, symbol: str) -> str:
        return f"{TopicBuilder.DOMAIN_MARKET}.{TopicBuilder.CAT_EQUITY}.{TopicBuilder.PART_BAR}.{timeframe}.{TopicBuilder._normalize_symbol(symbol)}"

    @staticmethod
    def for_equity_indicator_ema(timeframe: str, symbol: str) -> str:
        return f"{TopicBuilder.DOMAIN_MARKET}.{TopicBuilder.CAT_EQUITY}.{TopicBuilder.PART_INDICATOR}.{TopicBuilder.PART_EMA}.{timeframe}.{TopicBuilder._normalize_symbol(symbol)}"

    @staticmethod
    def for_equity_indicator_atr(timeframe: str, symbol: str) -> str:
        return f"{TopicBuilder.DOMAIN_MARKET}.{TopicBuilder.CAT_EQUITY}.{TopicBuilder.PART_INDICATOR}.{TopicBuilder.PART_ATR}.{timeframe}.{TopicBuilder._normalize_symbol(symbol)}"

    @staticmethod
    def for_equity_pivot(timeframe: str, symbol: str) -> str:
        return f"{TopicBuilder.DOMAIN_MARKET}.{TopicBuilder.CAT_EQUITY}.{TopicBuilder.PART_PIVOT}.{timeframe}.{TopicBuilder._normalize_symbol(symbol)}"

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

    # ── option market data — PRODUCER (the greeks server publishes these) ──────────
    # Hierarchical: md.option.<part>.UNDERLYING.EXPIRY.SIDE.STRIKE
    # Mirrors Scala forOptionTrade/forOptionQuote/forOptionGreeks + optionSuffix.
    # `side` is the single wire char 'C'/'P' (Scala CallPut.toChar).
    @staticmethod
    def for_option_trade(underlying: str, expiry: date, side: str, strike: Decimal | str | int | float) -> str:
        return f"{TopicBuilder.prefix_option_trades()}{TopicBuilder._option_suffix(underlying, expiry, side, strike)}"

    @staticmethod
    def for_option_quote(underlying: str, expiry: date, side: str, strike: Decimal | str | int | float) -> str:
        return f"{TopicBuilder.prefix_option_quotes()}{TopicBuilder._option_suffix(underlying, expiry, side, strike)}"

    @staticmethod
    def for_option_greeks(underlying: str, expiry: date, side: str, strike: Decimal | str | int | float) -> str:
        return f"{TopicBuilder.prefix_option_greeks()}{TopicBuilder._option_suffix(underlying, expiry, side, strike)}"

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

    # ── shared helpers (mirror Scala private normalizeSymbol / optionSuffix / format*) ──
    @staticmethod
    def _normalize_symbol(raw: str) -> str:
        """Scala normalizeSymbol: raw.trim.toUpperCase."""
        return raw.strip().upper()

    @staticmethod
    def _option_suffix(underlying: str, expiry: date, side: str, strike: Decimal | str | int | float) -> str:
        """UNDERLYING.EXPIRY.SIDE.STRIKE — Scala optionSuffix (TopicBuilder.scala optionSuffix)."""
        return (
            f"{TopicBuilder._normalize_symbol(underlying)}"
            f"{TopicBuilder.DOT}{TopicBuilder._format_expiry(expiry)}"
            f"{TopicBuilder.DOT}{TopicBuilder._format_side(side)}"
            f"{TopicBuilder.DOT}{TopicBuilder._format_strike(strike)}"
        )

    @staticmethod
    def _format_side(side: str) -> str:
        """Single wire char 'C'/'P' (Scala CallPut.toChar). Reject anything else at the boundary."""
        s = side.strip().upper()
        if s not in ("C", "P"):
            raise ValueError(f"option side must be 'C' or 'P', got {side!r}")
        return s

    @staticmethod
    def _format_expiry(expiry: date) -> str:
        """ISO local date, e.g. 2025-01-03 (Scala DateTimeFormatter.ISO_LOCAL_DATE)."""
        return expiry.isoformat()

    @staticmethod
    def _format_strike(strike: Decimal | str | int | float) -> str:
        """dollars_cents, e.g. 177.50 -> 177_50, 177.05 -> 177_05 (Scala formatStrike).

        strike * 100, rounded HALF_UP to an integer number of cents, then
        f"{cents // 100}_{cents % 100:02d}". Decimal + ROUND_HALF_UP is load-bearing:
        Python's built-in round() is banker's rounding and would disagree (e.g. 2.125).
        """
        cents = int((Decimal(str(strike)) * 100).to_integral_value(rounding=ROUND_HALF_UP))
        return f"{cents // 100}_{cents % 100:02d}"
