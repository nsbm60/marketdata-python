#!/usr/bin/env python3
"""
Shell optimizer — stub solver service for end-to-end wire format validation.

Accepts optimize requests per pc-optimization-spec-draft.md, returns plausible
stub responses. No real optimization; exercises downstream code paths.

Usage:
    python shell_optimizer.py

Environment variables:
    SHELL_OPTIMIZER_FORCE_STATUS    — optimal|infeasible|timeout|malformed_request|internal_error (default: optimal)
    SHELL_OPTIMIZER_SELECTION_COUNT — number of variables to select in optimal stub (default: 5)
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from discovery.service_locator import ServiceLocator
from service import Service

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)-5s [%(threadName)s] %(name)s - %(message)s",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
#
# Wire-format strings live in the shared wire_format module — one definition,
# used by both the shell and the real solver. Shell-specific config (env var
# names, selection default, solver name, ports) is defined below; these are
# not wire-format strings.

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
    F_ID,
    F_UNDERLYING,
    F_TYPE,
    F_DIRECTION,
    F_CURRENT_QUANTITY,
    F_THETA_PER_UNIT,
    F_SPREAD_COST_PER_UNIT,
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
    BCT_CONCENTRATION_LIMIT,
    BCT_DELTA_TARGET,
    DIAGNOSTIC_FIELDS,
)

# Shell-specific configuration (not wire-format strings)
ENV_FORCE_STATUS = "SHELL_OPTIMIZER_FORCE_STATUS"
ENV_SELECTION_COUNT = "SHELL_OPTIMIZER_SELECTION_COUNT"
DEFAULT_SELECTION_COUNT = 5

SOLVER_NAME = "shell"

# ─────────────────────────────────────────────────────────────
# Configuration (read once at import time)
# ─────────────────────────────────────────────────────────────

VALID_STATUSES = {STATUS_OPTIMAL, STATUS_INFEASIBLE, STATUS_TIMEOUT, STATUS_MALFORMED, STATUS_INTERNAL_ERROR}

FORCE_STATUS = os.environ.get(ENV_FORCE_STATUS, STATUS_OPTIMAL)
if FORCE_STATUS not in VALID_STATUSES:
    sys.exit(f"[Optimizer] Fatal: {ENV_FORCE_STATUS}='{FORCE_STATUS}' is not valid. Must be one of: {sorted(VALID_STATUSES)}")

try:
    SELECTION_COUNT = int(os.environ.get(ENV_SELECTION_COUNT, str(DEFAULT_SELECTION_COUNT)))
except ValueError:
    sys.exit(f"[Optimizer] Fatal: {ENV_SELECTION_COUNT}='{os.environ[ENV_SELECTION_COUNT]}' is not a valid integer.")

# Ports matching SystemConfig.OptimizerSettings
PUB_PORT = 6050
ROUTER_PORT = 6051

# ─────────────────────────────────────────────────────────────
# Handler
# ─────────────────────────────────────────────────────────────


def handle_optimize(payload: dict) -> dict:
    """Handle an optimize request. Returns the data portion of the response."""
    start = time.monotonic()

    # TODO(real-solver): Validate schema_version == SCHEMA_VERSION; reject mismatched versions with malformed_request.
    # Extract metadata
    request_id = payload[F_REQUEST_ID]
    policy_version = payload[F_POLICY_VERSION]
    request_timestamp = payload[F_TIMESTAMP]
    verbose = payload.get(F_VERBOSE, False)
    dvs = payload[F_DECISION_VARIABLES]

    log.info("[Optimizer] request_id=%s, %d decision_variables, force_status=%s",
             request_id, len(dvs), FORCE_STATUS)
    log.debug("[Optimizer] Full request: %s", json.dumps(payload))

    # Build echoed metadata
    meta = {
        F_SCHEMA_VERSION: SCHEMA_VERSION,
        F_REQUEST_ID: request_id,
        F_POLICY_VERSION: policy_version,
        F_REQUEST_TIMESTAMP: request_timestamp,
        F_RESPONSE_TIMESTAMP: datetime.now(timezone.utc).isoformat(),
    }

    # Dispatch on status
    if FORCE_STATUS == STATUS_OPTIMAL:
        response = {**meta, **_build_optimal(dvs, verbose, start)}
    elif FORCE_STATUS == STATUS_INFEASIBLE:
        response = {**meta, **_build_infeasible(dvs, start)}
    elif FORCE_STATUS == STATUS_TIMEOUT:
        response = {**meta, **_build_timeout(dvs, verbose, start)}
    elif FORCE_STATUS == STATUS_MALFORMED:
        response = {**meta, **_build_malformed()}
    elif FORCE_STATUS == STATUS_INTERNAL_ERROR:
        response = {**meta, **_build_internal_error()}
    else:
        raise RuntimeError(f"unreachable: FORCE_STATUS '{FORCE_STATUS}' validated at startup")

    log.debug("[Optimizer] Full response: %s", json.dumps(response))
    return response


# ─────────────────────────────────────────────────────────────
# Status-specific builders
# ─────────────────────────────────────────────────────────────


def _build_optimal(dvs: list, verbose: bool, start: float) -> dict:
    selected_ids = _select_targets(dvs)
    recommendations = _build_recommendations(dvs, selected_ids, verbose)
    score = _build_score(recommendations)
    return {
        F_STATUS: STATUS_OPTIMAL,
        F_RECOMMENDATIONS: recommendations,
        F_SCORE: score,
        F_BINDING_CONSTRAINTS: [],
        F_SOLVER_META: _build_solver_meta(STATUS_OPTIMAL, dvs, start),
    }


def _build_infeasible(dvs: list, start: float) -> dict:
    # Use first underlying from the request so the canned response is contextually plausible
    first_underlying = dvs[0][F_UNDERLYING] if dvs else "UNKNOWN"
    return {
        F_STATUS: STATUS_INFEASIBLE,
        F_BINDING_CONSTRAINTS: [
            {BC_TYPE: BCT_CONCENTRATION_LIMIT, BC_UNDERLYING: first_underlying, BC_LIMIT: 75000.0, BC_VALUE_AT_SOLUTION: None},
            {BC_TYPE: BCT_DELTA_TARGET, BC_UNDERLYING: first_underlying, BC_TARGET: 0, BC_TOLERANCE: 25, BC_VALUE_AT_SOLUTION: None},
        ],
        F_ERROR: "Shell-injected infeasibility for testing. Constraints would conflict given the request.",
        F_SOLVER_META: _build_solver_meta(STATUS_INFEASIBLE, dvs, start),
    }


def _build_timeout(dvs: list, verbose: bool, start: float) -> dict:
    selected_ids = _select_targets(dvs)
    recommendations = _build_recommendations(dvs, selected_ids, verbose)
    score = _build_score(recommendations)
    return {
        F_STATUS: STATUS_TIMEOUT,
        F_RECOMMENDATIONS: recommendations,
        F_SCORE: score,
        F_BINDING_CONSTRAINTS: [],
        F_ERROR: "Shell-injected timeout. Solution may not be optimal.",
        F_SOLVER_META: _build_solver_meta(STATUS_TIMEOUT, dvs, start, wall_clock_override=30.0),
    }


def _build_malformed() -> dict:
    return {
        F_STATUS: STATUS_MALFORMED,
        F_ERROR: "Shell-injected malformed request. Field 'policy' would have been rejected.",
        F_SOLVER_META: {SM_SOLVER_NAME: SOLVER_NAME},
    }


def _build_internal_error() -> dict:
    return {
        F_STATUS: STATUS_INTERNAL_ERROR,
        F_ERROR: "Shell-injected internal error. Solver crashed in a way that would normally be a bug.",
        F_SOLVER_META: {SM_SOLVER_NAME: SOLVER_NAME},
    }


# ─────────────────────────────────────────────────────────────
# Selection logic
# ─────────────────────────────────────────────────────────────


def _select_targets(dvs: list) -> set:
    """Select top-N decision variables by theta_per_unit > 0, return their ids."""
    eligible = [dv for dv in dvs if dv.get(F_THETA_PER_UNIT, 0) > 0]
    eligible.sort(key=lambda dv: dv[F_THETA_PER_UNIT], reverse=True)
    return {dv[F_ID] for dv in eligible[:SELECTION_COUNT]}


# ─────────────────────────────────────────────────────────────
# Recommendation and score construction
# ─────────────────────────────────────────────────────────────


def _build_recommendations(dvs: list, selected_ids: set, verbose: bool) -> list:
    recs = []
    for dv in dvs:
        dv_id = dv[F_ID]
        current_qty = dv.get(F_CURRENT_QUANTITY, 0)
        target_qty = 1 if dv_id in selected_ids else 0

        # Default: include if target > 0 or currently held
        include = target_qty > 0 or current_qty > 0
        if verbose:
            include = True
        if not include:
            continue

        theta = dv.get(F_THETA_PER_UNIT, 0) * target_qty
        spread = -(dv.get(F_SPREAD_COST_PER_UNIT, 0) * target_qty)
        total = theta + spread

        rec = {
            F_ID: dv_id,
            F_UNDERLYING: dv[F_UNDERLYING],
            F_TYPE: dv[F_TYPE],
            F_DIRECTION: dv[F_DIRECTION],
            F_CURRENT_QUANTITY: current_qty,
            F_TARGET_QUANTITY: target_qty,
            F_OBJECTIVE_CONTRIBUTION: {
                OC_TOTAL: total,
                OC_THETA: theta,
                OC_SPREAD_COST: spread,
                OC_DELTA_CEILING: 0,
                OC_GAMMA: 0,
            },
        }

        # Echo diagnostic fields if present
        for field in DIAGNOSTIC_FIELDS:
            if field in dv:
                rec[field] = dv[field]

        recs.append(rec)

    return recs


def _build_score(recommendations: list) -> dict:
    objective_value = sum(r[F_OBJECTIVE_CONTRIBUTION][OC_TOTAL] for r in recommendations)
    theta_total = sum(r[F_OBJECTIVE_CONTRIBUTION][OC_THETA] for r in recommendations)
    spread_total = sum(r[F_OBJECTIVE_CONTRIBUTION][OC_SPREAD_COST] for r in recommendations)

    return {
        SC_OBJECTIVE_VALUE: objective_value,
        SC_COMPONENTS: {
            SC_THETA_TOTAL: theta_total,
            SC_SPREAD_COST_TOTAL: spread_total,
            SC_DELTA_CEILING_PENALTY: 0,
            SC_GAMMA_PENALTY: 0,
        },
    }


def _build_solver_meta(status: str, dvs: list, start: float, wall_clock_override: float | None = None) -> dict:
    elapsed = time.monotonic() - start
    return {
        SM_SOLVER_NAME: SOLVER_NAME,
        SM_SOLVER_STATUS: status,
        SM_ITERATIONS: 0,
        SM_WALL_CLOCK_SECONDS: wall_clock_override if wall_clock_override is not None else round(elapsed, 4),
        SM_PROBLEM_SIZE: {
            SM_DECISION_VARIABLES: len(dvs),
            SM_CONSTRAINTS: 0,
        },
    }


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────


def main():
    log.info("[Optimizer] Shell optimizer starting (force_status=%s, selection_count=%d)",
             FORCE_STATUS, SELECTION_COUNT)

    service = Service(
        name=ServiceLocator.OPTIMIZER,
        pub_port=PUB_PORT,
        router_port=ROUTER_PORT,
    )
    service.register_handler(OP_OPTIMIZE, handle_optimize)
    service.run()


if __name__ == "__main__":
    main()
