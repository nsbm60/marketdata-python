"""
Phase 0, Problem 4: Multiple underlyings with per-underlying constraints.

Extends Problem 3 with a second underlying, and shows how per-underlying
constraints compose with portfolio-wide constraints.

- 5 options on NVDA, 4 options on AVGO (9 total)
- Per-underlying delta targets (NVDA neutral, AVGO slightly long)
- Per-underlying notional caps
- Portfolio-wide gamma floor, gross turnover, net cash outflow caps
"""

import numpy as np
import cvxpy as cp


def main():
    # --- Candidates: 9 options across 2 underlyings ---
    # Format: (underlying, label, delta, theta, gamma, price, strike)
    candidates = [
        # NVDA — spot ~= 200
        ("NVDA", "NVDA ATM-C (K=200)",  0.50, -10.0, 0.08, 3.50, 200.0),
        ("NVDA", "NVDA ATM-P (K=200)", -0.50,  -8.0, 0.08, 2.80, 200.0),
        ("NVDA", "NVDA OTM-C (K=210)",  0.30,  -5.0, 0.05, 1.25, 210.0),
        ("NVDA", "NVDA OTM-P (K=190)", -0.30,  -6.0, 0.05, 1.65, 190.0),
        ("NVDA", "NVDA Far-C (K=225)",  0.10,  -2.0, 0.02, 0.30, 225.0),
        # AVGO — spot ~= 180
        ("AVGO", "AVGO ATM-C (K=180)",  0.50,  -9.0, 0.07, 3.20, 180.0),
        ("AVGO", "AVGO ATM-P (K=180)", -0.50,  -7.0, 0.07, 2.60, 180.0),
        ("AVGO", "AVGO OTM-C (K=190)",  0.30,  -4.5, 0.04, 1.10, 190.0),
        ("AVGO", "AVGO OTM-P (K=170)", -0.30,  -5.5, 0.04, 1.50, 170.0),
    ]

    underlyings = [c[0] for c in candidates]
    labels      = [c[1] for c in candidates]
    deltas      = np.array([c[2] for c in candidates])
    thetas      = np.array([c[3] for c in candidates])
    gammas      = np.array([c[4] for c in candidates])
    prices      = np.array([c[5] for c in candidates])
    strikes     = np.array([c[6] for c in candidates])
    n = len(candidates)

    # --- Per-contract transaction costs ---
    transaction_cost_per_contract = np.full(n, 0.75)

    # --- Existing portfolio ---
    # Short 5 NVDA ATM puts, short 3 AVGO ATM puts — typical premium harvest
    current_qty = np.zeros(n)
    current_qty[3] = -5.0   # NVDA OTM-P
    current_qty[8] = -3.0   # AVGO OTM-P

    # --- Per-underlying configuration ---
    # NVDA: delta neutral, up to $500K notional
    # AVGO: delta slight long bias (+5), up to $300K notional
    per_underlying_config = {
    "NVDA": {"delta_target":  0, "delta_tolerance": 50, "max_notional": 500_000},
    "AVGO": {"delta_target": +5, "delta_tolerance": 50, "max_notional": 300_000},
    }
    MAX_GROSS_TURNOVER  = 20_000.0
    MAX_CASH_OUTFLOW    = 10_000.0
    GAMMA_FLOOR         = -10.0
    # --- Decision variable ---
    change = cp.Variable(n, name="change", integer=True)
    final_qty = current_qty + change

    # --- Derived quantities ---
    contract_values         = prices * 100
    notionals_per_contract  = strikes * 100

    gross_turnover    = cp.sum(cp.multiply(contract_values, cp.abs(change)))
    net_cash_outflow  = cp.sum(cp.multiply(contract_values, change))

    # --- Objective ---
    objective = cp.Maximize(
        thetas @ final_qty
        - transaction_cost_per_contract @ cp.abs(change)
    )

    # --- Constraints ---
    constraints = [
        gammas @ final_qty >= GAMMA_FLOOR,
        final_qty >= -10,
        final_qty <=  10,
        gross_turnover   <= MAX_GROSS_TURNOVER,
        net_cash_outflow <= MAX_CASH_OUTFLOW,
    ]

    # Per-underlying constraints (loop, one set per underlying)
    for u, cfg in per_underlying_config.items():
        mask = np.array([ul == u for ul in underlyings])

        # Delta target with tolerance
        u_delta = deltas[mask] @ final_qty[mask] * 100
        constraints.append(u_delta >= (cfg["delta_target"] - cfg["delta_tolerance"]))
        constraints.append(u_delta <= (cfg["delta_target"] + cfg["delta_tolerance"]))

        # Notional cap
        u_notional = notionals_per_contract[mask] @ cp.abs(final_qty[mask])
        constraints.append(u_notional <= cfg["max_notional"])

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.HIGHS)

    # --- Report ---
    print("=" * 110)
    print(f"Solver status: {prob.status}")
    print(f"Objective value (theta - costs): {prob.value:.2f}")
    print()
    print("Existing -> Final positions:")
    print(f"  {'Contract':<25} {'Price':>6} {'Strike':>7} {'Current':>8} "
          f"{'Change':>8} {'Final':>8} {'θ-contrib':>10} {'γ-contrib':>10}")
    print(f"  {'-'*25} {'-'*6} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*10}")
    for i in range(n):
        cur = current_qty[i]
        chg = change.value[i]
        fin = final_qty.value[i]
        t_c = thetas[i] * fin
        g_c = gammas[i] * fin
        print(f"  {labels[i]:<25} {prices[i]:>6.2f} {strikes[i]:>7.2f} "
              f"{cur:>8.2f} {chg:>+8.2f} {fin:>8.2f} "
              f"{t_c:>10.2f} {g_c:>10.3f}")

    print()
    print("Per-underlying breakdown:")
    print(f"  {'Underlying':<10} {'Delta':>8} {'Theta':>10} {'Gamma':>8} {'Notional':>12}")
    print(f"  {'-'*10} {'-'*8} {'-'*10} {'-'*8} {'-'*12}")
    for u, cfg in per_underlying_config.items():
        mask = np.array([ul == u for ul in underlyings])
        u_delta    = (deltas[mask] @ final_qty.value[mask]) * 100
        u_theta    = thetas[mask] @ final_qty.value[mask]
        u_gamma    = gammas[mask] @ final_qty.value[mask]
        u_notional = notionals_per_contract[mask] @ np.abs(final_qty.value[mask])
        print(f"  {u:<10} {u_delta:>8.2f} {u_theta:>10.2f} {u_gamma:>8.3f} "
              f"${u_notional:>10,.0f} "
              f"(delta target: {cfg['delta_target']:+d}±{cfg['delta_tolerance']}, "
              f"notional cap: ${cfg['max_notional']:,})")

    print()
    total_gross_turnover   = np.sum(np.abs(change.value) * contract_values)
    total_net_cash_outflow = np.sum(change.value * contract_values)
    total_transaction_cost = np.sum(np.abs(change.value) * transaction_cost_per_contract)
    net_theta              = thetas @ final_qty.value

    print(f"Portfolio totals:")
    print(f"  Net delta (both):     {(deltas @ final_qty.value * 100):>8.2f}")
    print(f"  Net gamma:            {(gammas @ final_qty.value):>8.3f}  (floor: {GAMMA_FLOOR})")
    print(f"  Net theta:            {net_theta:>8.2f}")
    print(f"  Gross turnover:       ${total_gross_turnover:>8,.2f}  (budget: ${MAX_GROSS_TURNOVER:,.0f})")
    print(f"  Net cash outflow:     ${total_net_cash_outflow:+,.2f}  (cap: ${MAX_CASH_OUTFLOW:+,.0f})")
    print(f"  Transaction costs:    ${total_transaction_cost:,.2f}")
    print(f"  Net theta after costs: {net_theta - total_transaction_cost:,.2f}")


if __name__ == "__main__":
    main()