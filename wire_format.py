"""Wire-format string constants for the optimizer protocol.

Single source of truth for every string that goes on the wire between
CalcServer and the optimizer service, per pc-optimization-spec-draft.md.
Used by both the shell optimizer and the real solver — one definition, not
two that can drift.

Constants only: no parsing helpers, no dataclasses, no response-building
logic. Just the strings (and the schema-version integer) that appear on the
wire. Service-specific values (e.g. the solver_name field's *value*, env var
names, ports) are not wire-format strings and live with their service.
"""

# Operation discriminator
OP_OPTIMIZE = "optimize"

# Schema version (value carried in both requests and responses)
SCHEMA_VERSION = 1

# Status enum values
STATUS_OPTIMAL = "optimal"
STATUS_INFEASIBLE = "infeasible"
STATUS_TIMEOUT = "timeout"
STATUS_MALFORMED = "malformed_request"
STATUS_INTERNAL_ERROR = "internal_error"

# Request top-level field names
F_SCHEMA_VERSION = "schema_version"
F_REQUEST_ID = "request_id"
F_POLICY_VERSION = "policy_version"
F_TIMESTAMP = "timestamp"
F_VERBOSE = "verbose"
F_DECISION_VARIABLES = "decision_variables"
F_PER_UNDERLYING = "per_underlying"
F_POLICY = "policy"
F_CAPITAL = "capital"

# Decision variable field names
F_ID = "id"
F_UNDERLYING = "underlying"
F_TYPE = "type"
F_DIRECTION = "direction"
F_CURRENT_QUANTITY = "current_quantity"
F_THETA_PER_UNIT = "theta_per_unit"
F_SPREAD_COST_PER_UNIT = "spread_cost_per_unit"
F_MARGIN_PER_UNIT = "margin_per_unit"
F_CLOSEABLE_ONLY = "closeable_only"
F_DELTA_PER_UNIT = "delta_per_unit"
F_GAMMA_PER_UNIT = "gamma_per_unit"
F_PREMIUM_PER_UNIT = "premium_per_unit"
F_STRIKE = "strike"
F_DAYS_TO_EXPIRY = "days_to_expiry"

# Decision variable `type` enum values
TYPE_OPTION = "option"
TYPE_EQUITY = "equity"

# Per-underlying record field names
F_SPOT = "spot"
F_DELTA_TARGET = "delta_target"
F_TOLERANCE = "tolerance"
F_MAX_POSITIONS = "max_positions"

# Policy field names
F_BP_LIMIT_PCT = "bp_limit_pct"
F_CONCENTRATION_LIMIT_PCT = "concentration_limit_pct"
F_YIELD_FLOOR_PCT = "yield_floor_pct"
F_DELTA_CEILING_THRESHOLD = "delta_ceiling_threshold"
F_LAMBDA_DELTA_CEILING = "lambda_delta_ceiling"
F_LAMBDA_GAMMA = "lambda_gamma"
F_LAMBDA_SPREAD = "lambda_spread"

# Capital field names
F_NET_LIQ = "net_liq"

# Response field names
F_REQUEST_TIMESTAMP = "request_timestamp"
F_RESPONSE_TIMESTAMP = "response_timestamp"
F_STATUS = "status"
F_RECOMMENDATIONS = "recommendations"
F_SCORE = "score"
F_BINDING_CONSTRAINTS = "binding_constraints"
F_ERROR = "error"
F_SOLVER_META = "solver_meta"
F_TARGET_QUANTITY = "target_quantity"
F_OBJECTIVE_CONTRIBUTION = "objective_contribution"

# Objective contribution component keys
OC_TOTAL = "total"
OC_THETA = "theta"
OC_SPREAD_COST = "spread_cost"
OC_DELTA_CEILING = "delta_ceiling"
OC_GAMMA = "gamma"

# Score keys
SC_OBJECTIVE_VALUE = "objective_value"
SC_COMPONENTS = "components"
SC_THETA_TOTAL = "theta_total"
SC_SPREAD_COST_TOTAL = "spread_cost_total"
SC_DELTA_CEILING_PENALTY = "delta_ceiling_penalty"
SC_GAMMA_PENALTY = "gamma_penalty"

# Solver meta keys
SM_SOLVER_NAME = "solver_name"
SM_SOLVER_STATUS = "solver_status"
SM_ITERATIONS = "iterations"
SM_WALL_CLOCK_SECONDS = "wall_clock_seconds"
SM_PROBLEM_SIZE = "problem_size"
SM_DECISION_VARIABLES = "decision_variables"
SM_CONSTRAINTS = "constraints"

# Binding constraints keys
BC_TYPE = "type"
BC_UNDERLYING = "underlying"
BC_LIMIT = "limit"
BC_VALUE_AT_SOLUTION = "value_at_solution"
BC_TARGET = "target"
BC_TOLERANCE = "tolerance"

# Binding constraints type discriminator values
BCT_BP_LIMIT = "bp_limit"
BCT_CONCENTRATION_LIMIT = "concentration_limit"
BCT_DELTA_TARGET = "delta_target"

# Diagnostic fields echoed from request to response recommendations
DIAGNOSTIC_FIELDS = ("strike", "expiry", "days_to_expiry", "iv")
