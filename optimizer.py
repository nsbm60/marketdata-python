#!/usr/bin/env python3
"""
Real optimizer service — CVXPY-based solver per pc-solver-spec-draft.md.

Same service identity as the shell (name `optimizer`, PUB 6050, ROUTER 6051,
`optimize` op, same response envelope). Replaces the shell's stub handler with
real optimization. Run the shell OR this — they bind the same ports.

    python optimizer.py

LAYER 3 (status handling): the full formulation from Layer 2 plus the complete
status taxonomy — optimal, infeasible, timeout, malformed_request,
internal_error — with per-status response shapes matching the optimization spec
(absent fields truly absent). binding_constraints is populated from the
request's hard constraints for infeasible, and is an empty list for
optimal/timeout (dual-based detection is a deferred future enhancement).

RUN-LOOP AMENDMENTS (optimizer-run-loop-spec-draft.md §3, 2026-10-21):
  - execution cost on the trade: exec_cost_per_unit · |target − current|;
  - interim holding horizon h = min(days_to_expiry, horizon_days): theta × h,
    delta ceiling unscaled, gamma variance over each pair's overlapping horizon
    min(h_i, h_j), built as time slices (interim, like the horizon itself);
  - an instrument's long and short legs are netted (the split is degenerate);
  - rounding: floor every quantity that consumes buying power, nearest otherwise;
    then stock repairs any delta band the rounding broke, by the fewest shares that
    bring it back inside; then every hard
    constraint is re-checked, and a remaining breach is `infeasible`;
  - min_improvement: the current book is scored under the same objective and
    the response carries action hold|trade (never hold a book that breaks BP
    or the delta band);
  - the delta band is dollar delta as % of net liq, turned into a share band per
    underlying: (target_pct ± tolerance_pct)/100 · net_liq / spot. It is the one
    per-name limit (the concentration limit was retired, same measure);
  - per-underlying terms built from a sparse group-membership matrix.

Status boundary: request interpretation (field access + float conversions)
happens OUTSIDE the error-catch, so missing/ill-typed fields propagate to the
service library's {ok:false,error} wrapper — never miscategorized as
internal_error. The internal_error catch wraps only the solve and
post-processing, where NaN/solver failures actually arise.

Persistence: deferred. The handler does not write to ClickHouse — that
infrastructure (pc-persistence-draft.md) does not exist yet (post-Layer-4).

Environment variables:
    OPTIMIZER_SOLVER          — CVXPY backend: clarabel|scs|osqp (default: clarabel)
    OPTIMIZER_TIME_LIMIT_SEC  — hard solve time limit, seconds (default: 30)
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cvxpy as cp
import numpy as np
import scipy.sparse as sp

from discovery import ControlService
from service import Service

from wire_format import (
    OP_OPTIMIZE,
    SCHEMA_VERSION,
    STATUS_OPTIMAL,
    STATUS_INFEASIBLE,
    STATUS_TIMEOUT,
    STATUS_MALFORMED,
    STATUS_INTERNAL_ERROR,
    F_SCHEMA_VERSION,
    F_REQUEST_ID,
    F_POLICY_VERSION,
    F_TIMESTAMP,
    F_VERBOSE,
    F_DECISION_VARIABLES,
    F_PER_UNDERLYING,
    F_POLICY,
    F_CAPITAL,
    F_HORIZON_DAYS,
    F_ID,
    F_UNDERLYING,
    F_TYPE,
    F_DIRECTION,
    F_CURRENT_QUANTITY,
    F_THETA_PER_UNIT,
    F_EXEC_COST_PER_UNIT,
    F_MARGIN_PER_UNIT,
    F_CLOSEABLE_ONLY,
    F_DELTA_PER_UNIT,
    F_DAYS_TO_EXPIRY,
    F_GAMMA_PER_UNIT,
    F_SPOT,
    F_DELTA_TARGET_PCT,
    F_DELTA_TOLERANCE_PCT,
    F_MAX_POSITIONS,
    F_TRADEABLE,
    F_BP_LIMIT_PCT,
    F_DELTA_CEILING_THRESHOLD,
    F_LAMBDA_DELTA_CEILING,
    F_LAMBDA_GAMMA,
    F_LAMBDA_SPREAD,
    F_MIN_IMPROVEMENT,
    F_NET_LIQ,
    F_REQUEST_TIMESTAMP,
    F_RESPONSE_TIMESTAMP,
    F_STATUS,
    F_RECOMMENDATIONS,
    F_SCORE,
    F_BINDING_CONSTRAINTS,
    F_ERROR,
    F_SOLVER_META,
    F_TARGET_QUANTITY,
    F_OPTIMAL_QUANTITY,
    F_ACTION,
    ACTION_TRADE,
    ACTION_HOLD,
    TYPE_OPTION,
    TYPE_EQUITY,
    F_OBJECTIVE_CONTRIBUTION,
    OC_TOTAL,
    OC_THETA,
    OC_EXEC_COST,
    OC_DELTA_CEILING,
    OC_GAMMA,
    SC_OBJECTIVE_VALUE,
    SC_OPTIMAL_OBJECTIVE_VALUE,
    SC_CURRENT_OBJECTIVE_VALUE,
    SC_COMPONENTS,
    SC_THETA_TOTAL,
    SC_EXEC_COST_TOTAL,
    SC_DELTA_CEILING_PENALTY,
    SC_GAMMA_PENALTY,
    SM_SOLVER_NAME,
    SM_SOLVER_STATUS,
    SM_ITERATIONS,
    SM_WALL_CLOCK_SECONDS,
    SM_PROBLEM_SIZE,
    SM_DECISION_VARIABLES,
    SM_CONSTRAINTS,
    BC_TYPE,
    BC_UNDERLYING,
    BC_LIMIT,
    BC_VALUE_AT_SOLUTION,
    BC_TARGET_PCT,
    BC_TOLERANCE_PCT,
    BC_VALUE_PCT,
    BCT_BP_LIMIT,
    BCT_DELTA_TARGET,
    DIAGNOSTIC_FIELDS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s [%(threadName)s] %(name)s - %(message)s",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Service-specific configuration (not wire-format strings)
# ─────────────────────────────────────────────────────────────

ENV_SOLVER = "OPTIMIZER_SOLVER"
DEFAULT_SOLVER = "clarabel"
ENV_TIME_LIMIT = "OPTIMIZER_TIME_LIMIT_SEC"
DEFAULT_TIME_LIMIT_SEC = 30.0

_BACKENDS = {
    "clarabel": cp.CLARABEL,
    "scs": cp.SCS,
    "osqp": cp.OSQP,
}
SOLVER_NAME = os.environ.get(ENV_SOLVER, DEFAULT_SOLVER).lower()
if SOLVER_NAME not in _BACKENDS:
    sys.exit(f"[Solver] Fatal: {ENV_SOLVER}='{SOLVER_NAME}' is not valid. Must be one of: {sorted(_BACKENDS)}")
SOLVER_BACKEND = _BACKENDS[SOLVER_NAME]

try:
    SOLVE_TIME_LIMIT_SEC = float(os.environ.get(ENV_TIME_LIMIT, str(DEFAULT_TIME_LIMIT_SEC)))
except ValueError:
    sys.exit(f"[Solver] Fatal: {ENV_TIME_LIMIT}='{os.environ[ENV_TIME_LIMIT]}' is not a valid float.")

# Recovers the raw 0..1 delta from share-equivalent delta_per_unit (× multiplier).
DELTA_RAW_DIVISOR = 100.0

# Absolute slack when checking the CURRENT book against the hard constraints
# (D4: never hold a book that breaks one). Floating-point noise only — the
# current book's values are request constants, not solver output.
CONSTRAINT_CHECK_EPS = 1e-6

# Solver noise allowance when flooring: 1.9999999 is 2 contracts, not 1.
ROUNDING_EPS = 1e-6

# Operator-facing message for a book that still breaks a hard constraint after
# rounding and the stock repair (normally only a non-tradeable underlying).
ERR_POST_ROUNDING = "Rounded book breaks hard constraints after the stock repair: "

# CVXPY status → our status. Verified empirically: Clarabel reports "user_limit"
# when the time limit is hit (see tools/test_optimizer_units.py).
_CVXPY_OPTIMAL = {"optimal", "optimal_inaccurate"}
_CVXPY_INFEASIBLE = {"infeasible", "infeasible_inaccurate"}
_CVXPY_LIMIT = {"user_limit"}

# Operator-facing error messages (content, not wire-format keys). The infeasible
# message stays non-committal: without IIS extraction we do not know which
# constraints conflict, only that the set in play is jointly unsatisfiable.
ERR_INFEASIBLE = ("Constraints could not be simultaneously satisfied; see "
                  "binding_constraints for the constraints in play.")
ERR_TIMEOUT = ("Solver exceeded the time budget; the solution may not be optimal "
               "(best feasible solution returned if one was found).")

# Control port matching SystemConfig.OptimizerSettings (same as the shell).
# No pub port: the service connects its PUB out to the bus (resolved in run()).
ROUTER_PORT = 6051


# ─────────────────────────────────────────────────────────────
# Handler
# ─────────────────────────────────────────────────────────────


_CAPTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools", "captured_requests")


def _capture_request(payload: dict, request_id: str) -> None:
    """Persist the exact incoming request so a run can be replayed offline through
    pipeline_trace (tools/replay_optimizer_run.py). One JSON per request; the directory is
    gitignored. Best-effort — a capture failure is logged and never affects the solve."""
    try:
        os.makedirs(_CAPTURE_DIR, exist_ok=True)
        path = os.path.join(_CAPTURE_DIR, f"{request_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        log.info("[Solver] captured request to %s", path)
    except Exception as e:
        log.warning("[Solver] request capture failed (%s): %s", request_id, e)


def handle_optimize(payload: dict) -> dict:
    """Handle an optimize request. Returns the data portion of the response."""
    start = time.monotonic()

    # Echoed metadata — present on every status. Missing metadata propagates to
    # the library's {ok:false} wrapper (deeply malformed request).
    request_id = payload[F_REQUEST_ID]
    policy_version = payload[F_POLICY_VERSION]
    request_timestamp = payload[F_TIMESTAMP]
    meta = {
        F_SCHEMA_VERSION: SCHEMA_VERSION,
        F_REQUEST_ID: request_id,
        F_POLICY_VERSION: policy_version,
        F_REQUEST_TIMESTAMP: request_timestamp,
        F_RESPONSE_TIMESTAMP: datetime.now(timezone.utc).isoformat(),
    }

    # Step 2: schema_version validation (missing or != 1 → malformed_request).
    request_version = payload.get(F_SCHEMA_VERSION)
    if request_version != SCHEMA_VERSION:
        log.warning("[Solver] request_id=%s malformed: schema_version=%r (expected %d)",
                    request_id, request_version, SCHEMA_VERSION)
        return _malformed_response(
            meta, f"Unsupported schema_version {request_version!r}; this solver requires {SCHEMA_VERSION}.")

    verbose = payload.get(F_VERBOSE, False)
    dvs = payload[F_DECISION_VARIABLES]
    per_underlying = payload[F_PER_UNDERLYING]
    policy = payload[F_POLICY]
    capital = payload[F_CAPITAL]
    horizon_days = payload[F_HORIZON_DAYS]

    log.info("[Solver] request_id=%s, %d decision_variables, solver=%s",
             request_id, len(dvs), SOLVER_NAME)
    log.debug("[Solver] Full request: %s", json.dumps(payload))
    _capture_request(payload, request_id)

    # Interpret inputs OUTSIDE the internal_error catch: missing fields (KeyError)
    # and ill-typed fields (ValueError) propagate to the library, never becoming
    # internal_error. NaN survives float() and is caught later as a solve failure.
    groups = _group_by_underlying(dvs)
    bundle = _extract_inputs(dvs, per_underlying, policy, capital, groups, horizon_days)
    n = len(dvs)

    # Solve + post-process INSIDE the catch: NaN/solver crashes → internal_error.
    try:
        target, problem, n_constraints = _build_and_solve(n, bundle)
        our_status = _map_status(problem.status)
        log.info("[Solver] request_id=%s, solver_status=%s -> %s, wall_clock=%.4fs",
                 request_id, problem.status, our_status, time.monotonic() - start)

        if our_status == STATUS_INFEASIBLE:
            return _infeasible_response(meta, bundle, groups, problem, n, n_constraints, start)
        if our_status == STATUS_INTERNAL_ERROR:
            return _internal_error_response(meta, f"Solver returned status '{problem.status}'.")
        return _build_solved_response(
            meta, our_status, target.value, problem, dvs, bundle, groups, verbose, n, n_constraints, start)
    except Exception as e:  # genuine solve/post-process failure
        log.warning("[Solver] request_id=%s internal error: %s", request_id, e, exc_info=True)
        return _internal_error_response(meta, f"{type(e).__name__}: {e}")


# ─────────────────────────────────────────────────────────────
# Input interpretation (outside the internal_error catch)
# ─────────────────────────────────────────────────────────────


def _group_by_underlying(dvs: list) -> dict:
    """underlying -> list of indices into dvs. Validates underlying presence."""
    groups = {}
    for i, dv in enumerate(dvs):
        groups.setdefault(dv[F_UNDERLYING], []).append(i)
    return groups


def _horizon(dv: dict, horizon_days: float) -> float:
    """Record's holding horizon h. Options: min(days_to_expiry, horizon_days),
    with days_to_expiry required, fractional, and non-negative. Equity: horizon_days
    (its theta and gamma are 0, so h has no effect). Any other type is rejected."""
    kind = dv[F_TYPE]
    if kind == TYPE_OPTION:
        dte = float(dv[F_DAYS_TO_EXPIRY])  # missing → KeyError (malformed)
        if not dte >= 0.0:  # also rejects NaN
            raise ValueError(f"{dv[F_ID]}: days_to_expiry must be >= 0, got {dte!r}")
        return min(dte, horizon_days)
    if kind == TYPE_EQUITY:
        return horizon_days
    raise ValueError(f"{dv[F_ID]}: unknown type {kind!r} (expected {TYPE_OPTION!r} or {TYPE_EQUITY!r})")


def _group_matrix(rows: np.ndarray, values: np.ndarray, n_groups: int) -> sp.csr_matrix:
    """Sparse (n_groups × n) matrix with values[i] at (rows[i], i): row u of
    M @ q is Σ_(i in u) values[i]·q[i]."""
    n = len(values)
    return sp.csr_matrix((values, (rows, np.arange(n))), shape=(n_groups, n))


def _gamma_slice_matrix(rows: np.ndarray, h: np.ndarray, gamma: np.ndarray, n_u: int) -> sp.csr_matrix:
    """Gamma variance over overlapping horizons, as a sum of squares.

    Two positions are both held for min(h_i, h_j), so the per-underlying variance is
    Σ_i Σ_j γ_i q_i γ_j q_j · min(h_i, h_j). Sorting the distinct horizons
    t_1 < t_2 < … (t_0 = 0) turns that into Σ_k (t_k − t_(k−1)) · (Σ_(i: h_i ≥ t_k) γ_i q_i)².
    Row (u, k) of the returned matrix holds √(t_k − t_(k−1)) · γ_i for every record i of
    underlying u still held in slice k, so the penalty is ‖M q‖². Interim, like the
    horizon itself, until lifecycle valuation.
    """
    n = len(h)
    t = np.unique(h[h > 0.0])                       # sorted distinct positive horizons
    n_slices = t.size
    if n_slices == 0:
        return sp.csr_matrix((0, n))
    dt = np.diff(np.concatenate(([0.0], t)))
    held_slices = np.searchsorted(t, h, side="right")  # slices k with t_k <= h_i
    cols = np.repeat(np.arange(n), held_slices)
    starts = np.repeat(np.cumsum(held_slices) - held_slices, held_slices)
    ks = np.arange(cols.size) - starts
    vals = np.sqrt(dt[ks]) * gamma[cols]
    return sp.csr_matrix((vals, (rows[cols] * n_slices + ks, cols)), shape=(n_u * n_slices, n))


def _extract_inputs(dvs: list, per_underlying: list, policy: dict, capital: dict,
                    groups: dict, horizon_days) -> dict:
    """Validate and convert every numeric input the solver consumes, plus touch
    the required identity fields so any missing/ill-typed field raises here —
    outside the internal_error catch. Returns a bundle of Python/numpy values."""
    horizon = float(horizon_days)
    if not horizon > 0.0:
        raise ValueError(f"horizon_days must be > 0, got {horizon_days!r}")

    n = len(dvs)
    theta = np.empty(n)
    exec_cost = np.empty(n)
    delta = np.empty(n)
    gamma = np.empty(n)
    margin = np.empty(n)
    current_qty = np.empty(n)
    h = np.empty(n)
    closeable = np.zeros(n, dtype=bool)
    und = []
    for i, dv in enumerate(dvs):
        # Touch required identity fields (presence validation; echoed later).
        _ = (dv[F_ID], dv[F_TYPE], dv[F_DIRECTION])
        und.append(dv[F_UNDERLYING])
        theta[i] = float(dv[F_THETA_PER_UNIT])
        exec_cost[i] = float(dv[F_EXEC_COST_PER_UNIT])
        delta[i] = float(dv[F_DELTA_PER_UNIT])
        gamma[i] = float(dv[F_GAMMA_PER_UNIT])
        margin[i] = float(dv[F_MARGIN_PER_UNIT])
        current_qty[i] = float(dv[F_CURRENT_QUANTITY])
        h[i] = _horizon(dv, horizon)
        closeable[i] = bool(dv.get(F_CLOSEABLE_ONLY))

    thr = float(policy[F_DELTA_CEILING_THRESHOLD])
    delta_pen = np.maximum(0.0, np.abs(delta / DELTA_RAW_DIVISOR) - thr)

    net_liq = float(capital[F_NET_LIQ])
    if not net_liq > 0:
        raise ValueError(f"net_liq must be positive (the delta band is a % of it), got {net_liq!r}")
    pu_map = {rec[F_UNDERLYING]: rec for rec in per_underlying}
    pu = {}
    tradeable = {}
    for u in groups:
        rec = pu_map[u]  # missing per_underlying record for a traded underlying → KeyError (malformed)
        pu[u] = (
            float(rec[F_SPOT]),
            float(rec[F_DELTA_TARGET_PCT]),
            float(rec[F_DELTA_TOLERANCE_PCT]),
            int(rec[F_MAX_POSITIONS]),
        )
        if not pu[u][0] > 0:
            raise ValueError(f"{u}: spot must be positive, got {pu[u][0]!r}")
        if not pu[u][2] > 0:
            raise ValueError(f"{u}: delta_tolerance_pct must be positive, got {pu[u][2]!r}")
        tradeable[u] = bool(rec[F_TRADEABLE])

    # Equity legs per underlying: the stock that repairs a rounding-broken band.
    # Equity delta is ±1 per share by definition (pc-optimization-spec-draft.md).
    equity_legs = {}
    for i, dv in enumerate(dvs):
        if dv[F_TYPE] != TYPE_EQUITY:
            continue
        direction = dv[F_DIRECTION]
        expected = {"long": 1.0, "short": -1.0}.get(direction)
        if expected is None or delta[i] != expected:
            raise ValueError(f"{dv[F_ID]}: equity {direction!r} must have delta_per_unit {expected!r}, got {delta[i]!r}")
        legs = equity_legs.setdefault(dv[F_UNDERLYING], {})
        if direction in legs:
            raise ValueError(f"{dv[F_ID]}: more than one {direction} equity record for {dv[F_UNDERLYING]}")
        legs[direction] = i

    # Long/short pairs of the same instrument, by the wire id convention: the id is the
    # instrument plus a direction suffix (pc-optimization-spec-draft.md, `id`).
    stems = {}
    for i, dv in enumerate(dvs):
        for suffix in ("-long", "-short"):
            if dv[F_ID].endswith(suffix) and dv[F_DIRECTION] == suffix[1:]:
                stems.setdefault(dv[F_ID][: -len(suffix)], {})[suffix[1:]] = i
    pairs = [(legs["long"], legs["short"]) for legs in stems.values() if "long" in legs and "short" in legs]

    # Per-underlying vectors in group order; row index per record.
    und_names = list(groups)
    row_of = {u: k for k, u in enumerate(und_names)}
    rows = np.array([row_of[u] for u in und], dtype=int)
    n_u = len(und_names)

    return {
        "theta": theta, "exec_cost": exec_cost, "delta": delta, "gamma": gamma,
        "margin": margin, "current_qty": current_qty, "delta_pen": delta_pen,
        "h": h, "closeable": closeable, "und": und,
        "und_names": und_names, "tradeable": tradeable, "equity_legs": equity_legs,
        "pairs": pairs,
        "lam_spread": float(policy[F_LAMBDA_SPREAD]),
        "lam_delta": float(policy[F_LAMBDA_DELTA_CEILING]),
        "lam_gamma": float(policy[F_LAMBDA_GAMMA]),
        "min_improvement": float(policy[F_MIN_IMPROVEMENT]),
        "bp_limit": float(policy[F_BP_LIMIT_PCT]) / 100.0 * net_liq,
        "net_liq": net_liq,
        "pu": pu,
        # Vectorized per-underlying structure (§3.4)
        "G_delta": _group_matrix(rows, delta, n_u),
        "G_gamma_slices": _gamma_slice_matrix(rows, h, gamma, n_u),
        # Share band from the dollar band: (target_pct ∓ tolerance_pct)/100 · net_liq / spot.
        "u_lo": np.array([(pu[u][1] - pu[u][2]) / 100.0 * net_liq / pu[u][0] for u in und_names]),
        "u_hi": np.array([(pu[u][1] + pu[u][2]) / 100.0 * net_liq / pu[u][0] for u in und_names]),
    }


def _map_status(cvxpy_status: str) -> str:
    """Map a CVXPY problem status to our response status."""
    if cvxpy_status in _CVXPY_OPTIMAL:
        return STATUS_OPTIMAL
    if cvxpy_status in _CVXPY_INFEASIBLE:
        return STATUS_INFEASIBLE
    if cvxpy_status in _CVXPY_LIMIT:
        return STATUS_TIMEOUT
    # unbounded, solver_error, infeasible_or_unbounded, None, ... — shouldn't
    # happen with valid bounded inputs; signals bad data / solver trouble.
    return STATUS_INTERNAL_ERROR


# ─────────────────────────────────────────────────────────────
# Formulation
# ─────────────────────────────────────────────────────────────


def _build_and_solve(n: int, bundle: dict):
    """Build the QP from the pre-extracted bundle and solve. No field access or
    float conversion happens here — all of that is done in _extract_inputs.

    Maximize  (theta∘h)·q − λ_spread·exec·|q − current| − λ_delta·delta_pen·q
              − λ_gamma·‖M_slices · q‖²
    s.t.      q ≥ 0;  q[closeable] ≤ current[closeable];  margin·q ≤ bp_limit;
              per underlying: lo ≤ G_δ q ≤ hi (the dollar band in shares).
    """
    target = cp.Variable(n, nonneg=True)  # lower bound 0
    current = bundle["current_qty"]
    constraints = []

    # Upper bound on closeable_only records: target_i <= current_quantity_i
    closeable_idx = np.flatnonzero(bundle["closeable"])
    if closeable_idx.size:
        constraints.append(target[closeable_idx] <= current[closeable_idx])

    # Buying-power limit
    constraints.append(bundle["margin"] @ target <= bundle["bp_limit"])

    # Per-underlying delta band (vectorized), in shares
    net_delta = bundle["G_delta"] @ target  # share-equivalent, one entry per underlying
    constraints.append(net_delta >= bundle["u_lo"])
    constraints.append(net_delta <= bundle["u_hi"])

    # Objective over the holding horizon (§3.2): theta × h, exec cost on the
    # trade (one-off), delta ceiling unscaled, gamma over overlapping horizons.
    obj = (bundle["theta"] * bundle["h"]) @ target
    obj = obj - bundle["lam_spread"] * (bundle["exec_cost"] @ cp.abs(target - current))
    obj = obj - bundle["lam_delta"] * (bundle["delta_pen"] @ target)
    obj = obj - bundle["lam_gamma"] * cp.sum_squares(bundle["G_gamma_slices"] @ target)

    problem = cp.Problem(cp.Maximize(obj), constraints)
    problem.solve(solver=SOLVER_BACKEND, **_solve_kwargs())
    n_scalar_constraints = sum(int(c.size) for c in constraints)
    return target, problem, n_scalar_constraints


def _solve_kwargs() -> dict:
    """Solver options. The hard time limit is wired for Clarabel (other backends
    are configurable but don't get a time limit in v1)."""
    if SOLVER_NAME == "clarabel":
        return {"time_limit": SOLVE_TIME_LIMIT_SEC}
    return {}


def _net_pairs(target_value, bundle: dict):
    """Net each instrument's long and short legs into one side.

    A contract's long and short records (and a stock's) have exactly opposite delta, gamma
    and theta and the same execution cost, so the objective cannot tell holding 0.24 long
    and 1.90 short apart from 1.66 short: the split is degenerate and the solver may return
    either. Netting keeps every greek, never raises execution cost (|a| + |b| ≥ |a − b|) or
    the delta-ceiling penalty, and lowers margin, so the netted book is at least as good
    and satisfies every constraint the split did. None stays None."""
    if target_value is None:
        return None
    x = np.asarray(target_value, dtype=float).copy()
    for li, si in bundle["pairs"]:
        net = x[li] - x[si]
        x[li], x[si] = max(net, 0.0), max(-net, 0.0)
    return x


def _extract_targets(target_value, bundle: dict) -> list:
    """Round the continuous solution to whole units (options and equity alike).

    A quantity that consumes a Category 1 budget rounds toward zero (solver spec,
    handler step 8): every record with margin_per_unit > 0 consumes buying power, so
    rounding down can never breach BP. Records with no margin round to nearest.
    ROUNDING_EPS keeps solver noise (1.9999999) from losing a whole contract.
    The delta band is two-sided, so rounding can still break it; the stock repair
    and the re-check handle that."""
    n = len(bundle["margin"])
    if target_value is None:
        return [0] * n
    x = np.maximum(np.asarray(target_value, dtype=float), 0.0)
    rounded = np.where(bundle["margin"] > 0.0, np.floor(x + ROUNDING_EPS), np.floor(x + 0.5))
    return [int(v) for v in rounded]


def _repair_band_with_stock(q: list, bundle: dict) -> tuple:
    """For each tradeable underlying whose rounded book is outside its delta band, move
    the stock by the smallest whole number of shares that brings net delta back inside
    the band — not all the way to the target. The continuous optimum sat at the band
    edge because that was optimal; the repair undoes the rounding error, it does not
    overrule the solver. Stock carries exactly 1 delta per share. Underlyings already
    inside their band are left alone. Returns (q, repaired underlyings)."""
    q = list(q)
    repaired = []
    eps = CONSTRAINT_CHECK_EPS
    net_delta = bundle["G_delta"] @ np.asarray(q, dtype=float)
    for k, u in enumerate(bundle["und_names"]):
        if bundle["u_lo"][k] - eps <= net_delta[k] <= bundle["u_hi"][k] + eps:
            continue
        legs = bundle["equity_legs"].get(u, {})
        if not bundle["tradeable"][u] or "long" not in legs or "short" not in legs:
            continue  # no stock to repair with; the re-check reports it
        li, si = legs["long"], legs["short"]
        if net_delta[k] > bundle["u_hi"][k]:
            shift = -int(np.ceil(net_delta[k] - bundle["u_hi"][k] - eps))   # sell just enough
        else:
            shift = int(np.ceil(bundle["u_lo"][k] - net_delta[k] - eps))    # buy just enough
        shares = (q[li] - q[si]) + shift
        q[li], q[si] = max(shares, 0), max(-shares, 0)
        repaired.append(u)
        log.info("[Solver] stock repair: %s net delta %.4f outside [%.4f, %.4f]; stock moved %+d to %+d shares",
                 u, net_delta[k], bundle["u_lo"][k], bundle["u_hi"][k], shift, shares)
    return q, repaired


def _breached_constraints(bundle: dict, q) -> list:
    """Hard constraints (BP, delta band) that book q breaks, as binding_constraints
    records: BP with value_at_solution, the band with value_pct (% of net liq)."""
    q = np.asarray(q, dtype=float)
    eps = CONSTRAINT_CHECK_EPS
    out = []
    margin_used = float(bundle["margin"] @ q)
    if margin_used > bundle["bp_limit"] + eps:
        out.append({BC_TYPE: BCT_BP_LIMIT, BC_LIMIT: bundle["bp_limit"], BC_VALUE_AT_SOLUTION: margin_used})
    net_delta = bundle["G_delta"] @ q
    for k, u in enumerate(bundle["und_names"]):
        if not (bundle["u_lo"][k] - eps <= net_delta[k] <= bundle["u_hi"][k] + eps):
            spot, dt, tol, _maxpos = bundle["pu"][u]
            value_pct = float(spot * net_delta[k] / bundle["net_liq"] * 100.0)
            out.append({BC_TYPE: BCT_DELTA_TARGET, BC_UNDERLYING: u, BC_TARGET_PCT: dt,
                        BC_TOLERANCE_PCT: tol, BC_VALUE_PCT: value_pct})
    return out


# ─────────────────────────────────────────────────────────────
# Objective contribution attribution
# ─────────────────────────────────────────────────────────────


def _compute_contributions(bundle: dict, q) -> dict:
    """Per-position objective_contribution, decomposed by term, as index-aligned
    numpy arrays keyed by the OC_* component names.

    The gamma term attributes the overlap-form penalty λ·Σ_r Y_r², Y = M q (time
    slices, see _gamma_slice_matrix), by each position's share of every slice sum:

        gamma_i = −λ_gamma · q_i · (Mᵀ Y)_i
                = −λ_gamma · Σ_(slices k it is held in) Δt_k · γ_i q_i · S_k

    where S_k is the underlying's Σ γ q over the positions held in slice k. These sum
    across positions to −λ_gamma·‖Mq‖², the score's gamma_penalty. Recomputed from the
    q passed in, serving the ranking pass, the reporting pass and the current book.
    """
    q = np.asarray(q, dtype=float)
    theta_c = bundle["theta"] * bundle["h"] * q
    exec_c = -bundle["lam_spread"] * bundle["exec_cost"] * np.abs(q - bundle["current_qty"])
    dceil_c = -bundle["lam_delta"] * bundle["delta_pen"] * q
    m = bundle["G_gamma_slices"]
    gamma_c = -bundle["lam_gamma"] * q * (m.T @ (m @ q))
    return {
        OC_THETA: theta_c,
        OC_EXEC_COST: exec_c,
        OC_DELTA_CEILING: dceil_c,
        OC_GAMMA: gamma_c,
        OC_TOTAL: theta_c + exec_c + dceil_c + gamma_c,
    }


def _cardinality_filter(groups: dict, q: list, bundle: dict, ranking_contribs: dict) -> list:
    """Keep the top max_positions nonzero positions per underlying, zero the rest.
    Ranking uses contributions against the full pre-cardinality composition."""
    q = list(q)
    totals = ranking_contribs[OC_TOTAL]
    for u, idx in groups.items():
        max_pos = bundle["pu"][u][3]
        nonzero = [i for i in idx if q[i] > 0]
        if len(nonzero) <= max_pos:
            continue
        nonzero.sort(key=lambda i: totals[i], reverse=True)
        for i in nonzero[max_pos:]:
            q[i] = 0
        log.info("[Solver] cardinality: underlying %s had %d positions, kept top %d",
                 u, len(nonzero), max_pos)
    return q


def _book_meets_constraints(bundle: dict, q) -> bool:
    """True when book q satisfies every hard constraint: closeable_only bounds,
    BP and the delta band. Used on the CURRENT book
    to decide whether hold is allowed (never hold a book that breaks one)."""
    q = np.asarray(q, dtype=float)
    closeable = bundle["closeable"]
    if np.any(q[closeable] > bundle["current_qty"][closeable] + CONSTRAINT_CHECK_EPS):
        return False
    return not _breached_constraints(bundle, q)


def _book_margin(bundle: dict, q) -> float:
    """Buying power the book q consumes: margin_per_unit · q, the quantity the BP
    constraint and the re-check both use."""
    return float(bundle["margin"] @ np.asarray(q, dtype=float))


def _post_solve_pipeline(target_value, bundle: dict, groups: dict) -> dict:
    """The deterministic post-solve sequence as named, inspectable stages, so the real
    response path and the diagnostics (pipeline_trace) run the identical steps with no
    drift. Keys, in order: solve, netting, rounding, cap, repair. 'solve' and 'netting'
    carry the continuous book; the recommended integer book is 'rounding' onward, and
    'repair' is optimal_q. Insertion order is preserved (dict, py3.7+)."""
    n = len(bundle["margin"])
    cont = [0.0] * n if target_value is None else list(np.maximum(np.asarray(target_value, dtype=float), 0.0))
    netted = _net_pairs(target_value, bundle)
    rounded = _extract_targets(netted, bundle)
    ranking_contribs = _compute_contributions(bundle, rounded)
    carded = _cardinality_filter(groups, rounded, bundle, ranking_contribs)
    repaired, _repaired_names = _repair_band_with_stock(carded, bundle)
    return {
        "solve": cont,
        "netting": cont if netted is None else list(netted),
        "rounding": rounded,
        "cap": carded,
        "repair": repaired,
    }


def pipeline_trace(payload: dict) -> dict:
    """Replay a request through the solve and the post-solve pipeline and report, after
    each step, the BP consumed and the per-underlying net delta (shares and % of net liq).
    Makes visible which step adds margin or pushes a name outside its band. For
    tools/replay_optimizer_run.py and regression tests — no wire I/O."""
    dvs = payload[F_DECISION_VARIABLES]
    groups = _group_by_underlying(dvs)
    bundle = _extract_inputs(dvs, payload[F_PER_UNDERLYING], payload[F_POLICY],
                             payload[F_CAPITAL], groups, payload[F_HORIZON_DAYS])
    target, problem, _ = _build_and_solve(len(dvs), bundle)
    stages = _post_solve_pipeline(target.value, bundle, groups)
    und = bundle["und_names"]
    steps = []
    for name, q in stages.items():
        nd = bundle["G_delta"] @ np.asarray(q, dtype=float)
        steps.append({
            "step": name,
            "bp_used": _book_margin(bundle, q),
            "net_delta_shares": {u: float(nd[k]) for k, u in enumerate(und)},
            "net_delta_pct": {u: float(bundle["pu"][u][0] * nd[k] / bundle["net_liq"] * 100.0)
                              for k, u in enumerate(und)},
        })
    return {
        "status": _map_status(problem.status),
        "bp_limit": bundle["bp_limit"],
        "net_liq": bundle["net_liq"],
        "bands_pct": {u: {"target": bundle["pu"][u][1], "tol": bundle["pu"][u][2]} for u in und},
        "steps": steps,
    }


# ─────────────────────────────────────────────────────────────
# Response construction
# ─────────────────────────────────────────────────────────────


def _build_solved_response(meta, status, target_value, problem, dvs, bundle, groups,
                           verbose, n, n_constraints, start) -> dict:
    """Build the response for optimal or timeout status. For timeout with no
    feasible point, recommendations and score are omitted.

    After the cardinality filter the current book is scored under the same
    objective. Hold when optimal − current < min_improvement AND the current
    book meets every hard constraint; trade otherwise. On hold every record's
    target_quantity is its current_quantity; optimal_quantity always carries the
    solver's book."""
    solver_meta = _build_solver_meta(problem, n, n_constraints, start)

    if status == STATUS_TIMEOUT and target_value is None:
        return {
            **meta,
            F_STATUS: STATUS_TIMEOUT,
            F_BINDING_CONSTRAINTS: [],
            F_ERROR: ERR_TIMEOUT,
            F_SOLVER_META: solver_meta,
        }

    # Net long/short legs → round (step 8) → cardinality (step 9) → stock repair →
    # re-check (step 10). One inspectable pipeline, shared with pipeline_trace.
    stages = _post_solve_pipeline(target_value, bundle, groups)
    optimal_q = stages["repair"]
    breaches = _breached_constraints(bundle, optimal_q)
    if breaches:
        detail = "; ".join(
            f"{b.get(BC_UNDERLYING, 'portfolio')} {b[BC_TYPE]} (value "
            + (f"{b[BC_VALUE_PCT]:.4f}% of net liq" if BC_VALUE_PCT in b else f"{b[BC_VALUE_AT_SOLUTION]:.4f}") + ")"
            for b in breaches)
        log.warning("[Solver] post-rounding breach: %s", detail)
        return {
            **meta,
            F_STATUS: STATUS_INFEASIBLE,
            F_BINDING_CONSTRAINTS: breaches,
            F_ERROR: ERR_POST_ROUNDING + detail + ".",
            F_SOLVER_META: solver_meta,
        }
    optimal_contribs = _compute_contributions(bundle, optimal_q)

    current_q = [dv[F_CURRENT_QUANTITY] for dv in dvs]
    current_contribs = _compute_contributions(bundle, current_q)

    optimal_obj = float(optimal_contribs[OC_TOTAL].sum())
    current_obj = float(current_contribs[OC_TOTAL].sum())
    current_valid = _book_meets_constraints(bundle, current_q)
    hold = current_valid and (optimal_obj - current_obj < bundle["min_improvement"])
    action = ACTION_HOLD if hold else ACTION_TRADE
    log.info("[Solver] action=%s optimal=%.4f current=%.4f gap=%.4f min_improvement=%.4f current_valid=%s",
             action, optimal_obj, current_obj, optimal_obj - current_obj,
             bundle["min_improvement"], current_valid)

    target_q = current_q if hold else optimal_q
    target_contribs = current_contribs if hold else optimal_contribs

    response = {
        **meta,
        F_STATUS: status,
        F_ACTION: action,
        F_RECOMMENDATIONS: _build_recommendations(dvs, target_q, optimal_q, target_contribs, verbose),
        F_SCORE: _build_score(target_contribs, optimal_obj, current_obj),
        F_BINDING_CONSTRAINTS: [],  # dual-based detection deferred to shape-experiment phase
        F_SOLVER_META: solver_meta,
    }
    if status == STATUS_TIMEOUT:
        response[F_ERROR] = ERR_TIMEOUT
    return response


def _build_recommendations(dvs: list, target_q: list, optimal_q: list, contribs: dict,
                           verbose: bool) -> list:
    """One record per decision variable. Default scope: target_quantity > 0 OR
    current_quantity > 0 OR optimal_quantity > 0 (so a hold still shows what the
    solver would have opened). Verbose: every decision variable."""
    recs = []
    for i, dv in enumerate(dvs):
        current_qty = dv[F_CURRENT_QUANTITY]
        qi = target_q[i]
        oi = optimal_q[i]
        if not (qi > 0 or current_qty > 0 or oi > 0 or verbose):
            continue
        rec = {
            F_ID: dv[F_ID],
            F_UNDERLYING: dv[F_UNDERLYING],
            F_TYPE: dv[F_TYPE],
            F_DIRECTION: dv[F_DIRECTION],
            F_CURRENT_QUANTITY: current_qty,
            F_TARGET_QUANTITY: qi,
            F_OPTIMAL_QUANTITY: oi,
            F_OBJECTIVE_CONTRIBUTION: {
                OC_TOTAL: float(contribs[OC_TOTAL][i]),
                OC_THETA: float(contribs[OC_THETA][i]),
                OC_EXEC_COST: float(contribs[OC_EXEC_COST][i]),
                OC_DELTA_CEILING: float(contribs[OC_DELTA_CEILING][i]),
                OC_GAMMA: float(contribs[OC_GAMMA][i]),
            },
        }
        for field in DIAGNOSTIC_FIELDS:
            if field in dv:
                rec[field] = dv[field]
        recs.append(rec)
    return recs


def _build_score(contribs: dict, optimal_obj: float, current_obj: float) -> dict:
    """Score of the recommended book, summed from its per-position contributions
    (== sum over recommendations, since excluded positions contribute zero),
    plus both books' objectives so the optimal-vs-current gap is visible."""
    return {
        SC_OBJECTIVE_VALUE: float(contribs[OC_TOTAL].sum()),
        SC_OPTIMAL_OBJECTIVE_VALUE: optimal_obj,
        SC_CURRENT_OBJECTIVE_VALUE: current_obj,
        SC_COMPONENTS: {
            SC_THETA_TOTAL: float(contribs[OC_THETA].sum()),
            SC_EXEC_COST_TOTAL: float(contribs[OC_EXEC_COST].sum()),
            SC_DELTA_CEILING_PENALTY: float(contribs[OC_DELTA_CEILING].sum()),
            SC_GAMMA_PENALTY: float(contribs[OC_GAMMA].sum()),
        },
    }


def _request_constraints(bundle: dict, groups: dict) -> list:
    """The request's hard constraints as binding_constraints records, with the
    value (value_at_solution / value_pct) null (there is no solution for an infeasible problem). We
    list all constraints in play — without IIS extraction we cannot single out
    the conflicting subset."""
    records = [{
        BC_TYPE: BCT_BP_LIMIT,
        BC_LIMIT: bundle["bp_limit"],
        BC_VALUE_AT_SOLUTION: None,
    }]
    for u in groups:
        _spot, dt, tol, _maxpos = bundle["pu"][u]
        records.append({
            BC_TYPE: BCT_DELTA_TARGET,
            BC_UNDERLYING: u,
            BC_TARGET_PCT: dt,
            BC_TOLERANCE_PCT: tol,
            BC_VALUE_PCT: None,
        })
    return records


def _infeasible_response(meta, bundle, groups, problem, n, n_constraints, start) -> dict:
    return {
        **meta,
        F_STATUS: STATUS_INFEASIBLE,
        F_BINDING_CONSTRAINTS: _request_constraints(bundle, groups),
        F_ERROR: ERR_INFEASIBLE,
        F_SOLVER_META: _build_solver_meta(problem, n, n_constraints, start),
    }


def _malformed_response(meta, error_msg: str) -> dict:
    return {
        **meta,
        F_STATUS: STATUS_MALFORMED,
        F_ERROR: error_msg,
        F_SOLVER_META: {SM_SOLVER_NAME: SOLVER_NAME},
    }


def _internal_error_response(meta, error_msg: str) -> dict:
    return {
        **meta,
        F_STATUS: STATUS_INTERNAL_ERROR,
        F_ERROR: error_msg,
        F_SOLVER_META: {SM_SOLVER_NAME: SOLVER_NAME},
    }


def _build_solver_meta(problem, n_vars: int, n_constraints: int, start: float) -> dict:
    elapsed = time.monotonic() - start
    iterations = 0
    stats = getattr(problem, "solver_stats", None)
    if stats is not None and getattr(stats, "num_iters", None) is not None:
        iterations = int(stats.num_iters)
    return {
        SM_SOLVER_NAME: SOLVER_NAME,
        SM_SOLVER_STATUS: problem.status,  # raw CVXPY status, preserved for diagnosis
        SM_ITERATIONS: iterations,
        SM_WALL_CLOCK_SECONDS: round(elapsed, 4),
        SM_PROBLEM_SIZE: {
            SM_DECISION_VARIABLES: n_vars,
            SM_CONSTRAINTS: n_constraints,
        },
    }


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────


def main():
    log.info("[Solver] Real optimizer starting (solver=%s, time_limit=%ss)",
             SOLVER_NAME, SOLVE_TIME_LIMIT_SEC)

    service = Service(
        name=ControlService.OPTIMIZER,
        router_port=ROUTER_PORT,
    )
    service.register_handler(OP_OPTIMIZE, handle_optimize)
    service.run()


if __name__ == "__main__":
    main()
