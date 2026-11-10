"""Price selection: the option price is a two-sided quote mid only (no trade fallback); the spot
(select_spot) may be a trade."""

from __future__ import annotations

from greeks_server.price import PriceKind, quote_mid, select_spot
from greeks_server.wire import Quote, Trade


def test_quote_mid_takes_a_valid_two_sided_quote() -> None:
    m = quote_mid(Quote(1.0, 1.2, 2000))
    assert m is not None and m.kind is PriceKind.MID and m.value == 1.1 and m.ts_ms == 2000


def test_quote_mid_rejects_non_two_sided_quotes() -> None:
    assert quote_mid(None) is None
    assert quote_mid(Quote(None, 1.2, 2000)) is None          # one-sided (no bid)
    assert quote_mid(Quote(1.0, None, 2000)) is None          # one-sided (no ask)
    assert quote_mid(Quote(0.0, 0.05, 2000)) is None          # zero bid is not two-sided
    assert quote_mid(Quote(1.3, 1.2, 2000)) is None           # crossed
    assert quote_mid(Quote(1.2, 1.2, 2000)) is not None       # locked is valid


def test_quote_mid_has_no_trade_fallback() -> None:
    # There is no trade argument at all: a contract with only a trade has no option price here.
    assert not hasattr(quote_mid, "__wrapped__")              # (documents intent; quote_mid takes only a quote)


def test_select_spot_prefers_quote_then_newest_valid_trade() -> None:
    t = 5000
    assert select_spot(Quote(101.0, 101.2, 3000), [Trade(100.0, 4500)], t) is not None
    assert select_spot(Quote(101.0, 101.2, 3000), [Trade(100.0, 4500)], t).kind is PriceKind.MID
    s = select_spot(None, [Trade(99.0, 2000), Trade(100.0, 4500)], t)
    assert s is not None and s.kind is PriceKind.LAST and s.value == 100.0 and s.ts_ms == 4500


def test_select_spot_rejects_prices_after_T() -> None:
    t = 5000
    s = select_spot(None, [Trade(188.70, 6000), Trade(186.64, 4800)], t)   # 6000 > T, rejected
    assert s is not None and s.value == 186.64 and s.ts_ms == 4800
    assert select_spot(Quote(1.0, 1.2, 6000), [], t) is None               # quote after T, no trade
