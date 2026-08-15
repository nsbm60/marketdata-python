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

Status boundary: request interpretation (field access + float conversions)
happens OUTSIDE the error-catch, so missing/ill-typed fields propagate to the
service library's {ok:false,error} wrapper — never miscategorized as
internal_error. The internal_error catch wraps only the solve and
post-processing, where NaN/solver failures actually arise.

There is no yield floor: yield is a display measure, not an optimization
control. yield_floor_pct (if present in policy) is ignored.

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

from discovery.service_locator import ServiceLocator
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
    F_ID,
    F_UNDERLYING,
    F_TYPE,
    F_DIRECTION,
    F_CURRENT_QUANTITY,
    F_THETA_PER_UNIT,
    F_SPREAD_COST_PER_UNIT,
    F_MARGIN_PER_UNIT,
    F_CLOSEABLE_ONLY,
    F_DELTA_PER_UNIT,
    F_GAMMA_PER_UNIT,
    F_SPOT,
    F_DELTA_TARGET,
    F_TOLERANCE,
    F_MAX_POSITIONS,
    F_BP_LIMIT_PCT,
    F_CONCENTRATION_LIMIT_PCT,
    F_DELTA_CEILING_THRESHOLD,
    F_LAMBDA_DELTA_CEILING,
    F_LAMBDA_GAMMA,
    F_LAMBDA_SPREAD,
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
    F_OBJECTIVE_CONTRIBUTION,
    OC_TOTAL,
    OC_THETA,
    OC_SPREAD_COST,
    OC_DELTA_CEILING,
    OC_GAMMA,
    SC_OBJECTIVE_VALUE,
    SC_COMPONENTS,
    SC_THETA_TOTAL,
    SC_SPREAD_COST_TOTAL,
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
    BC_TARGET,
    BC_TOLERANCE,
    BCT_BP_LIMIT,
    BCT_CONCENTRATION_LIMIT,
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

# Ports matching SystemConfig.OptimizerSettings (same as the shell)
PUB_PORT = 6050
ROUTER_PORT = 6051


# ─────────────────────────────────────────────────────────────
# Handler
# ─────────────────────────────────────────────────────────────


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

    log.info("[Solver] request_id=%s, %d decision_variables, solver=%s",
             request_id, len(dvs), SOLVER_NAME)
    log.debug("[Solver] Full request: %s", json.dumps(payload))

    # Interpret inputs OUTSIDE the internal_error catch: missing fields (KeyError)
    # and ill-typed fields (ValueError) propagate to the library, never becoming
    # internal_error. NaN survives float() and is caught later as a solve failure.
    groups = _group_by_underlying(dvs)
    bundle = _extract_inputs(dvs, per_underlying, policy, capital, groups)
    n = len(dvs)

    # Solve + post-process INSIDE the catch: NaN/solver crashes → internal_error.
    try:
        target, problem, n_constraints = _build_and_solve(n, bundle, groups)
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


def _extract_inputs(dvs: list, per_underlying: list, policy: dict, capital: dict, groups: dict) -> dict:
    """Validate and convert every numeric input the solver consumes, plus touch
    the required identity fields so any missing/ill-typed field raises here —
    outside the internal_error catch. Returns a bundle of Python/numpy values."""
    n = len(dvs)
    theta = np.empty(n)
    spread = np.empty(n)
    delta = np.empty(n)
    gamma = np.empty(n)
    margin = np.empty(n)
    current_qty = np.empty(n)
    closeable = []
    und = []
    for i, dv in enumerate(dvs):
        # Touch required identity fields (presence validation; echoed later).
        _ = (dv[F_ID], dv[F_TYPE], dv[F_DIRECTION])
        und.append(dv[F_UNDERLYING])
        theta[i] = float(dv[F_THETA_PER_UNIT])
        spread[i] = float(dv[F_SPREAD_COST_PER_UNIT])
        delta[i] = float(dv[F_DELTA_PER_UNIT])
        gamma[i] = float(dv[F_GAMMA_PER_UNIT])
        margin[i] = float(dv[F_MARGIN_PER_UNIT])
        current_qty[i] = float(dv[F_CURRENT_QUANTITY])
        closeable.append(bool(dv.get(F_CLOSEABLE_ONLY)))

    thr = float(policy[F_DELTA_CEILING_THRESHOLD])
    delta_pen = np.maximum(0.0, np.abs(delta / DELTA_RAW_DIVISOR) - thr)

    net_liq = float(capital[F_NET_LIQ])
    pu_map = {rec[F_UNDERLYING]: rec for rec in per_underlying}
    pu = {}
    for u in groups:
        rec = pu_map[u]  # missing per_underlying record for a traded underlying → KeyError (malformed)
        pu[u] = (
            float(rec[F_SPOT]),
            float(rec[F_DELTA_TARGET]),
            float(rec[F_TOLERANCE]),
            int(rec[F_MAX_POSITIONS]),
        )

    return {
        "theta": theta, "spread": spread, "delta": delta, "gamma": gamma,
        "margin": margin, "current_qty": current_qty, "delta_pen": delta_pen,
        "closeable": closeable, "und": und,
        "lam_spread": float(policy[F_LAMBDA_SPREAD]),
        "lam_delta": float(policy[F_LAMBDA_DELTA_CEILING]),
        "lam_gamma": float(policy[F_LAMBDA_GAMMA]),
        "bp_limit": float(policy[F_BP_LIMIT_PCT]) / 100.0 * net_liq,
        "conc_limit": float(policy[F_CONCENTRATION_LIMIT_PCT]) / 100.0 * net_liq,
        "pu": pu,
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


def _build_and_solve(n: int, bundle: dict, groups: dict):
    """Build the QP from the pre-extracted bundle and solve. No field access or
    float conversion happens here — all of that is done in _extract_inputs."""
    target = cp.Variable(n, nonneg=True)  # lower bound 0
    constraints = []

    # Upper bound on closeable_only records: target_i <= current_quantity_i
    for i in range(n):
        if bundle["closeable"][i]:
            constraints.append(target[i] <= bundle["current_qty"][i])

    # Buying-power limit
    constraints.append(bundle["margin"] @ target <= bundle["bp_limit"])

    # Per-underlying concentration and delta-target constraints
    delta = bundle["delta"]
    for u, idx in groups.items():
        spot, dt, tol, _maxpos = bundle["pu"][u]
        net_delta = sum(delta[i] * target[i] for i in idx)  # share-equivalent
        constraints.append(cp.abs(spot * net_delta) <= bundle["conc_limit"])
        constraints.append(net_delta >= dt - tol)
        constraints.append(net_delta <= dt + tol)

    # Objective: theta − λ_spread·spread − λ_delta·delta_penalty − λ_gamma·gamma²(per underlying)
    gamma = bundle["gamma"]
    obj = bundle["theta"] @ target
    obj = obj - bundle["lam_spread"] * (bundle["spread"] @ target)
    obj = obj - bundle["lam_delta"] * (bundle["delta_pen"] @ target)
    for u, idx in groups.items():
        group_gamma = sum(gamma[i] * target[i] for i in idx)
        obj = obj - bundle["lam_gamma"] * cp.square(group_gamma)

    problem = cp.Problem(cp.Maximize(obj), constraints)
    problem.solve(solver=SOLVER_BACKEND, **_solve_kwargs())
    return target, problem, len(constraints)


def _solve_kwargs() -> dict:
    """Solver options. The hard time limit is wired for Clarabel (other backends
    are configurable but don't get a time limit in v1)."""
    if SOLVER_NAME == "clarabel":
        return {"time_limit": SOLVE_TIME_LIMIT_SEC}
    return {}


def _extract_targets(target_value, n: int) -> list:
    """Round the continuous solution to integers (options and equity alike).
    Clears sub-tolerance solver noise on degenerate variables. None → all zero."""
    if target_value is None:
        return [0] * n
    return [max(0, int(round(float(target_value[i])))) for i in range(n)]


# ─────────────────────────────────────────────────────────────
# Objective contribution attribution
# ─────────────────────────────────────────────────────────────


def _compute_contributions(bundle: dict, groups: dict, q: list) -> list:
    """Per-position objective_contribution, decomposed by term, index-aligned.

    The gamma term uses the group-sum attribution from expanding
    (Σ x_i)² = Σ x_i·(Σ_j x_j) with x_i = gamma_i·q_i:

        gamma_i = −λ_gamma · (gamma_i·q_i) · (Σ_{j in U(i)} gamma_j·q_j)

    NOT (gamma_i·q_i)². Per-position gamma terms sum across a group to
    −λ_gamma·(Σ gamma·q)² and across groups to the score's gamma_penalty. The
    group sum is recomputed from the q passed in, serving both the ranking pass
    (rounded q, full composition) and the reporting pass (post-cardinality q).
    """
    theta, spread, delta_pen, gamma = bundle["theta"], bundle["spread"], bundle["delta_pen"], bundle["gamma"]
    lam_spread, lam_delta, lam_gamma = bundle["lam_spread"], bundle["lam_delta"], bundle["lam_gamma"]
    und = bundle["und"]

    group_gamma_sum = {u: sum(gamma[i] * q[i] for i in idx) for u, idx in groups.items()}

    contribs = []
    for i in range(len(q)):
        theta_c = float(theta[i] * q[i])
        spread_c = float(-lam_spread * spread[i] * q[i])
        dceil_c = float(-lam_delta * delta_pen[i] * q[i])
        gamma_c = float(-lam_gamma * (gamma[i] * q[i]) * group_gamma_sum[und[i]])
        contribs.append({
            OC_THETA: theta_c,
            OC_SPREAD_COST: spread_c,
            OC_DELTA_CEILING: dceil_c,
            OC_GAMMA: gamma_c,
            OC_TOTAL: theta_c + spread_c + dceil_c + gamma_c,
        })
    return contribs


def _cardinality_filter(groups: dict, q: list, bundle: dict, ranking_contribs: list) -> list:
    """Keep the top max_positions nonzero positions per underlying, zero the rest.
    Ranking uses contributions against the full pre-cardinality composition."""
    q = list(q)
    for u, idx in groups.items():
        max_pos = bundle["pu"][u][3]
        nonzero = [i for i in idx if q[i] > 0]
        if len(nonzero) <= max_pos:
            continue
        nonzero.sort(key=lambda i: ranking_contribs[i][OC_TOTAL], reverse=True)
        for i in nonzero[max_pos:]:
            q[i] = 0
        log.info("[Solver] cardinality: underlying %s had %d positions, kept top %d",
                 u, len(nonzero), max_pos)
    return q


# ─────────────────────────────────────────────────────────────
# Response construction
# ─────────────────────────────────────────────────────────────


def _build_solved_response(meta, status, target_value, problem, dvs, bundle, groups,
                           verbose, n, n_constraints, start) -> dict:
    """Build the response for optimal or timeout status. For timeout with no
    feasible point, recommendations and score are omitted."""
    solver_meta = _build_solver_meta(problem, n, n_constraints, start)

    if status == STATUS_TIMEOUT and target_value is None:
        return {
            **meta,
            F_STATUS: STATUS_TIMEOUT,
            F_BINDING_CONSTRAINTS: [],
            F_ERROR: ERR_TIMEOUT,
            F_SOLVER_META: solver_meta,
        }

    rounded = _extract_targets(target_value, n)
    ranking_contribs = _compute_contributions(bundle, groups, rounded)
    final_q = _cardinality_filter(groups, rounded, bundle, ranking_contribs)
    final_contribs = _compute_contributions(bundle, groups, final_q)

    response = {
        **meta,
        F_STATUS: status,
        F_RECOMMENDATIONS: _build_recommendations(dvs, final_q, final_contribs, verbose),
        F_SCORE: _build_score(final_contribs),
        F_BINDING_CONSTRAINTS: [],  # dual-based detection deferred to shape-experiment phase
        F_SOLVER_META: solver_meta,
    }
    if status == STATUS_TIMEOUT:
        response[F_ERROR] = ERR_TIMEOUT
    return response


def _build_recommendations(dvs: list, q: list, contribs: list, verbose: bool) -> list:
    """One record per decision variable. Default scope: target_quantity > 0 OR
    current_quantity > 0. Verbose: every decision variable."""
    recs = []
    for i, dv in enumerate(dvs):
        current_qty = dv[F_CURRENT_QUANTITY]
        qi = q[i]
        if not (qi > 0 or current_qty > 0 or verbose):
            continue
        c = contribs[i]
        rec = {
            F_ID: dv[F_ID],
            F_UNDERLYING: dv[F_UNDERLYING],
            F_TYPE: dv[F_TYPE],
            F_DIRECTION: dv[F_DIRECTION],
            F_CURRENT_QUANTITY: current_qty,
            F_TARGET_QUANTITY: qi,
            F_OBJECTIVE_CONTRIBUTION: {
                OC_TOTAL: c[OC_TOTAL],
                OC_THETA: c[OC_THETA],
                OC_SPREAD_COST: c[OC_SPREAD_COST],
                OC_DELTA_CEILING: c[OC_DELTA_CEILING],
                OC_GAMMA: c[OC_GAMMA],
            },
        }
        for field in DIAGNOSTIC_FIELDS:
            if field in dv:
                rec[field] = dv[field]
        recs.append(rec)
    return recs


def _build_score(contribs: list) -> dict:
    """Score summed from the realized per-position contributions (== sum over
    recommendations, since excluded positions contribute zero)."""
    return {
        SC_OBJECTIVE_VALUE: sum(c[OC_TOTAL] for c in contribs),
        SC_COMPONENTS: {
            SC_THETA_TOTAL: sum(c[OC_THETA] for c in contribs),
            SC_SPREAD_COST_TOTAL: sum(c[OC_SPREAD_COST] for c in contribs),
            SC_DELTA_CEILING_PENALTY: sum(c[OC_DELTA_CEILING] for c in contribs),
            SC_GAMMA_PENALTY: sum(c[OC_GAMMA] for c in contribs),
        },
    }


def _request_constraints(bundle: dict, groups: dict) -> list:
    """The request's hard constraints as binding_constraints records, with
    value_at_solution null (there is no solution for an infeasible problem). We
    list all constraints in play — without IIS extraction we cannot single out
    the conflicting subset."""
    records = [{
        BC_TYPE: BCT_BP_LIMIT,
        BC_LIMIT: bundle["bp_limit"],
        BC_VALUE_AT_SOLUTION: None,
    }]
    for u in groups:
        spot, dt, tol, _maxpos = bundle["pu"][u]
        records.append({
            BC_TYPE: BCT_CONCENTRATION_LIMIT,
            BC_UNDERLYING: u,
            BC_LIMIT: bundle["conc_limit"],
            BC_VALUE_AT_SOLUTION: None,
        })
        records.append({
            BC_TYPE: BCT_DELTA_TARGET,
            BC_UNDERLYING: u,
            BC_TARGET: dt,
            BC_TOLERANCE: tol,
            BC_VALUE_AT_SOLUTION: None,
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
        name=ServiceLocator.OPTIMIZER,
        pub_port=PUB_PORT,
        router_port=ROUTER_PORT,
    )
    service.register_handler(OP_OPTIMIZE, handle_optimize)
    service.run()


if __name__ == "__main__":
    main()
