"""Read-only probe: compare our option greeks/IV (trading.option_trade) with Massive's Historical
Greeks & IV endpoint.

    python -m greeks.validation.massive_analytics <date> <underlying> [<date> <underlying> …] [--csv PATH]

Nothing is written to ClickHouse. The two methods differ by design (expected, not errors): Massive
prices the NBBO mid at the snapshot time with a binomial American model, a continuous dividend yield
and a Treasury-curve rate; we invert each trade with Black-76 European, a discrete-dividend forward
and SOFR (1-month T-bill pre-2018). Greek conventions DO match (greeks/solver/black76.py: theta per
calendar day, vega/rho per one point, delta/gamma in spot terms), so theta/gamma/vega are compared
directly.

Sync httpx only (no asyncio/threading under greeks/). The fetch and every comparison/bucket/ATM
function is pure and injectable, so the tests touch no HTTP or ClickHouse.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Mapping, Optional, Sequence
from zoneinfo import ZoneInfo

import httpx
import numpy as np

from greeks.ch import get_ch_client
from greeks.config import get_config
from greeks.domain import Dividend
from greeks.occ import parse_occ
from ml.shared.calendar import session_close_et

MASSIVE_BASE = "https://api.massive.com"
ANALYTICS_PATH = "/options/v1/analytics"
_ET = ZoneInfo("America/New_York")

SNAPSHOT_BEFORE_CLOSE_MIN = 15   # S = session close − 15 min (15:45 ET, 12:45 on half days)
WINDOW_MIN = 5                   # our trades within S ± 5 minutes
MIN_OUR_TRADES = 3               # need at least this many of our trades to compare a contract
ATM_DELTA_LO, ATM_DELTA_HI = 0.25, 0.75
ATM_MIN_CONTRACTS = 3            # and strikes straddling the forward

_DELTA_BUCKETS = ("|d|<0.10", "0.10-0.25", "0.25-0.75", "0.75-0.90", "|d|>0.90")
_DTE_BUCKETS = ("0-2", "3-7", "8-30", "31-90", ">90")
# IV/greeks compared per bucket; each carries (median signed, median abs, p90 abs).
_METRICS = ("iv", "delta", "gamma", "theta", "vega")


# ───────────────────────── Massive rows ─────────────────────────

@dataclass(frozen=True)
class MassiveRow:
    """One /options/v1/analytics row. Greeks/IV may be null (expected); the structural fields must be
    present or the reply is bad."""
    ticker: str
    contract_type: str          # "call" | "put"
    strike_price: float
    expiration_date: date
    date: date
    underlying_price: float
    option_price: Optional[float]
    bid: Optional[float]
    ask: Optional[float]
    dividend_yield: Optional[float]
    risk_free_rate: Optional[float]
    model: Optional[str]
    implied_volatility: Optional[float]
    delta: Optional[float]
    gamma: Optional[float]
    theta: Optional[float]
    vega: Optional[float]
    rho: Optional[float]
    open_interest: Optional[float]
    volume: Optional[float]

    @property
    def side(self) -> str:
        return "C" if self.contract_type.lower().startswith("c") else "P"

    @property
    def dte(self) -> int:
        return (self.expiration_date - self.date).days


def _f(v: Any) -> Optional[float]:
    if v is None or isinstance(v, bool):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _req_float(obj: Mapping[str, Any], key: str) -> float:
    v = _f(obj.get(key))
    if v is None:
        raise ValueError(f"Massive analytics row missing/!numeric field {key!r}: {obj}")
    return v


def _req_date(obj: Mapping[str, Any], key: str) -> date:
    v = obj.get(key)
    if not isinstance(v, str):
        raise ValueError(f"Massive analytics row missing/!string date {key!r}: {obj}")
    return date.fromisoformat(v[:10])


def parse_row(obj: Mapping[str, Any]) -> MassiveRow:
    """Parse one analytics row. Raises ValueError (a bad reply) if a structural field is absent."""
    ticker = obj.get("ticker")
    ctype = obj.get("contract_type")
    if not isinstance(ticker, str) or not isinstance(ctype, str):
        raise ValueError(f"Massive analytics row missing ticker/contract_type: {obj}")
    return MassiveRow(
        ticker=ticker, contract_type=ctype,
        strike_price=_req_float(obj, "strike_price"),
        expiration_date=_req_date(obj, "expiration_date"),
        date=_req_date(obj, "date"),
        underlying_price=_req_float(obj, "underlying_price"),
        option_price=_f(obj.get("option_price")), bid=_f(obj.get("bid")), ask=_f(obj.get("ask")),
        dividend_yield=_f(obj.get("dividend_yield")), risk_free_rate=_f(obj.get("risk_free_rate")),
        model=obj.get("model") if isinstance(obj.get("model"), str) else None,
        implied_volatility=_f(obj.get("implied_volatility")),
        delta=_f(obj.get("delta")), gamma=_f(obj.get("gamma")), theta=_f(obj.get("theta")),
        vega=_f(obj.get("vega")), rho=_f(obj.get("rho")),
        open_interest=_f(obj.get("open_interest")), volume=_f(obj.get("volume")))


def fetch_analytics(underlying: str, day: date, api_key: str, *,
                    client: Optional[httpx.Client] = None,
                    timeout_s: float = 60.0, max_attempts: int = 5, backoff_s: float = 1.0
                    ) -> list[MassiveRow]:
    """All analytics rows for (underlying, day), every page. Raises with the status and body on an
    HTTP error, or ValueError on a reply that isn't the expected shape — never partial output."""
    if not api_key:
        raise ValueError("MASSIVE_API_KEY is not set — this probe reads the Massive analytics endpoint")
    params: Optional[dict[str, Any]] = {
        "underlying_ticker": underlying.upper(), "date": day.isoformat(), "limit": 50000,
        "apiKey": api_key,
    }
    url: Optional[str] = f"{MASSIVE_BASE}{ANALYTICS_PATH}"
    own = client is None
    http = client or httpx.Client(timeout=timeout_s)
    out: list[MassiveRow] = []
    try:
        while url is not None:
            body = _get_with_retry(http, url, params, api_key, max_attempts, backoff_s)
            results = body.get("results")
            if not isinstance(results, list):
                raise ValueError(f"Massive analytics reply has no results list: {_clip(body)}")
            for row in results:
                if not isinstance(row, Mapping):
                    raise ValueError(f"Massive analytics result is not an object: {row}")
                out.append(parse_row(row))
            nxt = body.get("next_url")
            url = nxt if isinstance(nxt, str) and nxt else None
            params = None
    finally:
        if own:
            http.close()
    return out


def _get_with_retry(http: httpx.Client, url: str, params: Optional[Mapping[str, Any]],
                    api_key: str, max_attempts: int, backoff_s: float) -> dict[str, Any]:
    last: Optional[BaseException] = None
    for attempt in range(max_attempts):
        try:
            if params is not None:
                resp = http.get(url, params=dict(params))
            else:  # a next_url — re-append the key as the other Massive tools do
                resp = http.get(url if "apiKey=" in url else f"{url}{'&' if '?' in url else '?'}apiKey={api_key}")
            if resp.status_code in (429, 500, 502, 503, 504):
                last = httpx.HTTPStatusError(f"retryable {resp.status_code}", request=resp.request, response=resp)
                time.sleep(backoff_s * (2 ** attempt))
                continue
            if resp.status_code >= 400:
                raise RuntimeError(f"Massive analytics HTTP {resp.status_code}: {_clip(resp.text)}")
            data = resp.json()
            if not isinstance(data, dict):
                raise ValueError(f"Massive analytics JSON must be an object: {_clip(data)}")
            return data
        except (httpx.TransportError, httpx.TimeoutException) as e:
            last = e
            time.sleep(backoff_s * (2 ** attempt))
    raise RuntimeError(f"Massive analytics request failed after {max_attempts} attempts: {last}")


def _clip(v: Any) -> str:
    s = v if isinstance(v, str) else repr(v)
    return s if len(s) <= 2000 else s[:2000] + "…"


# ───────────────────────── our side ─────────────────────────

@dataclass(frozen=True)
class OurTrade:
    iv: Optional[float]
    delta: Optional[float]
    gamma: Optional[float]
    theta: Optional[float]
    vega: Optional[float]
    spot: Optional[float]
    discount: Optional[float]
    tte: Optional[float]
    forward: Optional[float]


@dataclass(frozen=True)
class OurStats:
    n: int
    iv: Optional[float]
    delta: Optional[float]
    gamma: Optional[float]
    theta: Optional[float]
    vega: Optional[float]
    spot: Optional[float]
    forward: Optional[float]
    tte: Optional[float]
    rate: Optional[float]       # implied: -ln(discount)/tte
    div_yield: Optional[float]  # implied: rate - ln(F/S)/tte


def snapshot_window_utc(day: date) -> tuple[datetime, datetime]:
    """The S ± WINDOW_MIN window in UTC, where S = the session close − SNAPSHOT_BEFORE_CLOSE_MIN."""
    close = session_close_et(day)
    s_et = datetime.combine(day, close, tzinfo=_ET) - timedelta(minutes=SNAPSHOT_BEFORE_CLOSE_MIN)
    lo = (s_et - timedelta(minutes=WINDOW_MIN)).astimezone(timezone.utc)
    hi = (s_et + timedelta(minutes=WINDOW_MIN)).astimezone(timezone.utc)
    return lo, hi


def read_our_trades(ch: Any, underlying: str, day: date) -> dict[str, list[OurTrade]]:
    """Our successful v2 trade-basis greeks for (underlying, day) in the S ± 5-minute window, one
    query, grouped by bare-OSI symbol. Read-only."""
    lo, hi = snapshot_window_utc(day)
    res = ch.query(
        "SELECT symbol, iv, delta, gamma, theta, vega, toFloat64(spot_at_trade), "
        "greeks_discount, greeks_tte, greeks_forward "
        "FROM trading.option_trade FINAL "
        "WHERE underlying = %(u)s AND session_date = %(d)s "
        "  AND greeks_status = 'success' AND greeks_methodology = 'v2' "
        "  AND trade_ts BETWEEN %(lo)s AND %(hi)s",
        parameters={"u": underlying.upper(), "d": day, "lo": lo, "hi": hi})
    by: dict[str, list[OurTrade]] = {}
    for r in res.result_rows:
        by.setdefault(str(r[0]), []).append(OurTrade(
            iv=_f(r[1]), delta=_f(r[2]), gamma=_f(r[3]), theta=_f(r[4]), vega=_f(r[5]),
            spot=_f(r[6]), discount=_f(r[7]), tte=_f(r[8]), forward=_f(r[9])))
    return by


def our_stats(trades: Sequence[OurTrade]) -> OurStats:
    iv = _median([t.iv for t in trades])
    spot = _median([t.spot for t in trades])
    fwd = _median([t.forward for t in trades])
    tte = _median([t.tte for t in trades])
    disc = _median([t.discount for t in trades])
    rate = implied_rate(disc, tte)
    return OurStats(
        n=sum(1 for t in trades if t.iv is not None),
        iv=iv, delta=_median([t.delta for t in trades]), gamma=_median([t.gamma for t in trades]),
        theta=_median([t.theta for t in trades]), vega=_median([t.vega for t in trades]),
        spot=spot, forward=fwd, tte=tte, rate=rate, div_yield=implied_div(rate, fwd, spot, tte))


def implied_rate(discount: Optional[float], tte: Optional[float]) -> Optional[float]:
    """r from the discount factor exp(-r·T): r = -ln(discount)/T."""
    if discount is None or tte is None or tte <= 0 or discount <= 0:
        return None
    return float(-np.log(discount) / tte)


def implied_div(rate: Optional[float], forward: Optional[float], spot: Optional[float],
                tte: Optional[float]) -> Optional[float]:
    """q from the forward F = S·exp((r-q)·T): q = r - ln(F/S)/T."""
    if rate is None or forward is None or spot is None or tte is None or tte <= 0 or forward <= 0 or spot <= 0:
        return None
    return float(rate - np.log(forward / spot) / tte)


# ───────────────────────── join + buckets ─────────────────────────

@dataclass(frozen=True)
class JoinedRow:
    session_date: date
    underlying: str
    ticker: str
    side: str
    strike: float
    expiry: date
    dte: int
    ex_div: bool
    bucket_delta: str
    bucket_dte: str
    reason: str                 # "matched" | "null_greeks" | "too_few_trades"
    massive_iv: Optional[float]
    massive_delta: Optional[float]
    massive_gamma: Optional[float]
    massive_theta: Optional[float]
    massive_vega: Optional[float]
    massive_underlying_price: Optional[float]
    massive_rfr: Optional[float]
    massive_div_yield: Optional[float]
    our_n: int
    our_iv: Optional[float]
    our_delta: Optional[float]
    our_gamma: Optional[float]
    our_theta: Optional[float]
    our_vega: Optional[float]
    our_spot: Optional[float]
    our_rate: Optional[float]
    our_div_yield: Optional[float]

    def diff(self, metric: str) -> Optional[float]:
        o = getattr(self, f"our_{metric}")
        m = getattr(self, f"massive_{metric}")
        return (o - m) if (o is not None and m is not None) else None


def bucket_delta(abs_delta: Optional[float]) -> str:
    if abs_delta is None:
        return "no_delta"
    if abs_delta < 0.10:
        return _DELTA_BUCKETS[0]
    if abs_delta < 0.25:
        return _DELTA_BUCKETS[1]
    if abs_delta <= 0.75:
        return _DELTA_BUCKETS[2]
    if abs_delta <= 0.90:
        return _DELTA_BUCKETS[3]
    return _DELTA_BUCKETS[4]


def bucket_dte(dte: int) -> str:
    if dte <= 2:
        return _DTE_BUCKETS[0]
    if dte <= 7:
        return _DTE_BUCKETS[1]
    if dte <= 30:
        return _DTE_BUCKETS[2]
    if dte <= 90:
        return _DTE_BUCKETS[3]
    return _DTE_BUCKETS[4]


def ex_div_between(divs: Sequence[Dividend], day: date, expiry: date) -> bool:
    return any(day < d.ex_date <= expiry for d in divs)


@dataclass(frozen=True)
class JoinResult:
    rows: list[JoinedRow]              # all Massive rows (matched + unmatched), for the CSV
    matched: list[JoinedRow]           # reason == "matched"
    n_too_few: int
    n_null_greeks: int


def join(massive: Sequence[MassiveRow], ours: Mapping[str, list[OurTrade]], day: date,
         divs: Sequence[Dividend]) -> JoinResult:
    rows: list[JoinedRow] = []
    n_too_few = n_null = 0
    for mr in massive:
        osi = parse_occ(mr.ticker).symbol
        st = our_stats(ours.get(osi, []))
        ex = ex_div_between(divs, day, mr.expiration_date)

        def mk(reason: str, mr: MassiveRow = mr, osi: str = osi, st: OurStats = st, ex: bool = ex) -> JoinedRow:
            return JoinedRow(
                session_date=day, underlying=osi, ticker=mr.ticker, side=mr.side, strike=mr.strike_price,
                expiry=mr.expiration_date, dte=mr.dte, ex_div=ex,
                bucket_delta=bucket_delta(abs(mr.delta) if mr.delta is not None else None),
                bucket_dte=bucket_dte(mr.dte), reason=reason,
                massive_iv=mr.implied_volatility, massive_delta=mr.delta, massive_gamma=mr.gamma,
                massive_theta=mr.theta, massive_vega=mr.vega, massive_underlying_price=mr.underlying_price,
                massive_rfr=mr.risk_free_rate, massive_div_yield=mr.dividend_yield, our_n=st.n,
                our_iv=st.iv, our_delta=st.delta, our_gamma=st.gamma, our_theta=st.theta, our_vega=st.vega,
                our_spot=st.spot, our_rate=st.rate, our_div_yield=st.div_yield)

        if mr.implied_volatility is None:
            n_null += 1
            rows.append(mk("null_greeks"))
        elif st.n < MIN_OUR_TRADES:
            n_too_few += 1
            rows.append(mk("too_few_trades"))
        else:
            rows.append(mk("matched"))
    matched = [r for r in rows if r.reason == "matched"]
    return JoinResult(rows=rows, matched=matched, n_too_few=n_too_few, n_null_greeks=n_null)


# ───────────────────────── stats + ATM fit ─────────────────────────

def _median(xs: Sequence[Optional[float]]) -> Optional[float]:
    vals = [x for x in xs if x is not None]
    return float(np.median(vals)) if vals else None


def _p90_abs(xs: Sequence[float]) -> Optional[float]:
    return float(np.percentile([abs(x) for x in xs], 90)) if xs else None


@dataclass(frozen=True)
class Triple:
    signed: Optional[float]
    abs_: Optional[float]
    p90: Optional[float]


def triple(diffs: Sequence[Optional[float]]) -> Triple:
    vals = [d for d in diffs if d is not None]
    if not vals:
        return Triple(None, None, None)
    return Triple(float(np.median(vals)), float(np.median([abs(v) for v in vals])), _p90_abs(vals))


def bucket_stats(rows: Sequence[JoinedRow]) -> dict[str, Triple]:
    return {m: triple([r.diff(m) for r in rows]) for m in _METRICS}


def atm_line_iv(massive: Sequence[MassiveRow], expiry: date, forward: float) -> Optional[tuple[float, int]]:
    """Massive's ATM IV: the unweighted least-squares line of IV on k = ln(K/F) through the expiry's
    |delta| 0.25–0.75 contracts, read at k = 0 (the intercept). Needs ≥3 contracts straddling the
    forward (min k < 0 < max k); else None."""
    ks: list[float] = []
    ivs: list[float] = []
    for mr in massive:
        if mr.expiration_date != expiry or mr.implied_volatility is None or mr.delta is None:
            continue
        if not (ATM_DELTA_LO <= abs(mr.delta) <= ATM_DELTA_HI):
            continue
        if forward <= 0 or mr.strike_price <= 0:
            continue
        ks.append(float(np.log(mr.strike_price / forward)))
        ivs.append(mr.implied_volatility)
    if len(ks) < ATM_MIN_CONTRACTS or min(ks) >= 0 or max(ks) <= 0:
        return None
    slope, intercept = np.polyfit(np.array(ks), np.array(ivs), 1)
    return float(intercept), len(ks)


# ───────────────────────── ClickHouse reads (read-only) ─────────────────────────

@dataclass(frozen=True)
class AtmRow:
    expiry: date
    dte: int
    atm_iv: float
    forward: float
    n_strikes: int


def read_iv_term(ch: Any, underlying: str, day: date) -> list[AtmRow]:
    res = ch.query(
        "SELECT expiry, atm_iv, forward, n_strikes FROM trading.iv_term_daily FINAL "
        "WHERE method = 'atm_v1' AND status = 'ok' AND underlying = %(u)s AND session_date = %(d)s "
        "ORDER BY expiry",
        parameters={"u": underlying.upper(), "d": day})
    out: list[AtmRow] = []
    for r in res.result_rows:
        exp = r[0]
        out.append(AtmRow(expiry=exp, dte=(exp - day).days, atm_iv=float(r[1]),
                          forward=float(r[2]), n_strikes=int(r[3])))
    return out


def read_dividends(ch: Any, underlying: str, day: date) -> list[Dividend]:
    res = ch.query(
        "SELECT symbol, ex_date, amount FROM trading.dividend "
        "WHERE symbol = %(u)s AND ex_date > %(d)s ORDER BY ex_date",
        parameters={"u": underlying.upper(), "d": day})
    return [Dividend(underlying=str(r[0]), amount=float(r[2]), ex_date=r[1]) for r in res.result_rows]


# ───────────────────────── rendering ─────────────────────────

def _fmt(x: Optional[float], prec: int = 4) -> str:
    return f"{x:+.{prec}f}" if x is not None else "   —   "


def _render_bucket_section(title: str, groups: Sequence[tuple[str, list[JoinedRow]]]) -> list[str]:
    lines = [f"  {title}"]
    hdr = f"    {'bucket':12s} {'n':>5s}"
    for m in _METRICS:
        hdr += f"  {m + ':sgn':>9s} {'abs':>8s} {'p90':>8s}"
    lines.append(hdr)
    for label, rows in groups:
        if not rows:
            continue
        stats = bucket_stats(rows)
        line = f"    {label:12s} {len(rows):>5d}"
        for m in _METRICS:
            t = stats[m]
            line += f"  {_fmt(t.signed):>9s} {_fmt(t.abs_, 4).lstrip('+'):>8s} {_fmt(t.p90, 4).lstrip('+'):>8s}"
        lines.append(line)
    return lines


def render_buckets(title: str, matched: Sequence[JoinedRow]) -> list[str]:
    lines = [title, "  (metric columns: ours − Massive; sgn=median signed, abs=median |·|, p90=p90 |·|; IV & vega in vol points)"]
    lines += _render_bucket_section(
        "by |delta|", [(b, [r for r in matched if r.bucket_delta == b]) for b in _DELTA_BUCKETS])
    lines += _render_bucket_section(
        "by DTE", [(b, [r for r in matched if r.bucket_dte == b]) for b in _DTE_BUCKETS])
    lines += _render_bucket_section(
        "by side", [(s, [r for r in matched if r.side == s]) for s in ("C", "P")])
    lines += _render_bucket_section(
        "ex-dividend before expiry", [(f"exdiv {s}", [r for r in matched if r.ex_div and r.side == s]) for s in ("C", "P")])
    return lines


def render_inputs(matched: Sequence[JoinedRow], n_too_few: int, n_null: int) -> list[str]:
    spot = _median([_sub(r.our_spot, r.massive_underlying_price) for r in matched])
    rate = _median([_sub(r.our_rate, r.massive_rfr) for r in matched])
    div = _median([_sub(r.our_div_yield, r.massive_div_yield) for r in matched])
    return [
        "  inputs (median ours − Massive across matched contracts):",
        f"    spot:          {_fmt(spot, 4)}   (ours spot_at_trade vs underlying_price)",
        f"    rate:          {_fmt(rate, 5)}   (ours -ln(discount)/T vs risk_free_rate)",
        f"    dividend yield:{_fmt(div, 5)}   (ours r - ln(F/S)/T vs dividend_yield)",
        f"    matched={len(matched)}  too_few_trades={n_too_few}  null_greeks={n_null}",
    ]


def _sub(a: Optional[float], b: Optional[float]) -> Optional[float]:
    return (a - b) if (a is not None and b is not None) else None


def render_atm(atm: Sequence[AtmRow], massive: Sequence[MassiveRow]) -> list[str]:
    lines = ["  ATM IV (ours = iv_term_daily atm_v1; Massive = IV-vs-k line at k=0):",
             f"    {'expiry':10s} {'dte':>4s} {'ours':>8s} {'massive':>8s} {'diff':>8s} {'n_ours':>6s} {'n_mas':>6s}"]
    for a in atm:
        fit = atm_line_iv(massive, a.expiry, a.forward)
        if fit is None:
            lines.append(f"    {a.expiry.isoformat():10s} {a.dte:>4d} {a.atm_iv:>8.4f} {'—':>8s} {'—':>8s} {a.n_strikes:>6d} {0:>6d}")
        else:
            miv, nmas = fit
            lines.append(f"    {a.expiry.isoformat():10s} {a.dte:>4d} {a.atm_iv:>8.4f} {miv:>8.4f} "
                         f"{a.atm_iv - miv:>+8.4f} {a.n_strikes:>6d} {nmas:>6d}")
    return lines


# ───────────────────────── CSV ─────────────────────────

def write_csv(path: str, rows: Sequence[JoinedRow]) -> None:
    cols = list(asdict(rows[0]).keys()) if rows else [f.name for f in JoinedRow.__dataclass_fields__.values()]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


# ───────────────────────── orchestration ─────────────────────────

def run_pair(ch: Any, api_key: str, day: date, underlying: str, *,
             client: Optional[httpx.Client] = None) -> JoinResult:
    """Fetch, read our side, join. Prints the per-pair report. Returns the join for the overall roll-up."""
    massive = fetch_analytics(underlying, day, api_key, client=client)
    ours = read_our_trades(ch, underlying, day)
    divs = read_dividends(ch, underlying, day)
    jr = join(massive, ours, day, divs)
    atm = read_iv_term(ch, underlying, day)

    print(f"\n===== {underlying.upper()} {day.isoformat()} — {len(massive)} Massive rows =====")
    for line in render_inputs(jr.matched, jr.n_too_few, jr.n_null_greeks):
        print(line)
    for line in render_buckets("  bucket residuals (matched contracts):", jr.matched):
        print(line)
    for line in render_atm(atm, massive):
        print(line)
    return jr


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="greeks.validation.massive_analytics")
    p.add_argument("pairs", nargs="+", help="<date> <underlying> [<date> <underlying> …]")
    p.add_argument("--csv", default=None, help="write the contract-level joined rows to this path")
    args = p.parse_args(argv)
    if len(args.pairs) % 2 != 0:
        p.error("arguments must be <date> <underlying> pairs")
    pairs = [(date.fromisoformat(args.pairs[i]), args.pairs[i + 1]) for i in range(0, len(args.pairs), 2)]

    api_key = get_config().api_keys.massive_api_key
    if not api_key:
        print("ERROR: MASSIVE_API_KEY is not set (the Massive analytics endpoint needs it)", file=sys.stderr)
        return 1

    ch = get_ch_client()
    all_matched: list[JoinedRow] = []
    all_rows: list[JoinedRow] = []
    try:
        for day, underlying in pairs:
            jr = run_pair(ch, api_key, day, underlying)
            all_matched.extend(jr.matched)
            all_rows.extend(jr.rows)
    except Exception as exc:  # noqa: BLE001 — a fetch/reply failure aborts the run with its message
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("\n===== overall (all pairs) =====")
    for line in render_buckets("  bucket residuals (matched contracts):", all_matched):
        print(line)
    if args.csv:
        write_csv(args.csv, all_rows)
        print(f"\nwrote {len(all_rows)} contract rows to {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
