"""
Phase 0, Problem 2: Add gamma constraint.

Same 5 options as Problem 1, but now with gamma. Constraint:
net portfolio gamma must be >= -5 (don't let gamma get too negative).

This prevents the LP from loading up on short premium with no regard
for how fast delta will drift when underlying moves.

Expected: smaller short positions, perhaps avoidance of high-gamma options,
lower theta but a safer portfolio.
"""

import numpy as np
import cvxpy as cp


def main():
    # Option contracts with realistic Greeks.
    # Note: ATM options have higher gamma than OTM. Far-OTM has very low gamma.
    # theta tends to be roughly proportional to gamma for short-dated options.

    deltas = np.array([0.50, -0.50, 0.30, -0.30, 0.10])
    thetas = np.array([-10.0, -8.0, -5.0, -6.0, -2.0])
    gammas = np.array([0.08, 0.08, 0.05, 0.05, 0.02])  # positive for long, negative for short
    n = len(deltas)

    labels = [
        "ATM-Call",
        "ATM-Put",
        "OTM-Call",
        "OTM-Put",
        "Far-OTM-Call",
    ]

    # Decision variable
    quantity = cp.Variable(n, name="quantity")

    # Objective: maximize theta
    objective = cp.Maximize(thetas @ quantity)

    # Gamma cap: net portfolio gamma must be >= -5
    # For a short position, gamma contribution is negative
    # Setting gamma >= -5 means "don't be shorter than -5 gamma"
    GAMMA_FLOOR = -1.5

    constraints = [
        deltas @ quantity == 0,
        gammas @ quantity >= GAMMA_FLOOR,
        quantity >= -10,
        quantity <= 10,
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.HIGHS)

    print("=" * 70)
    print(f"Solver status: {prob.status}")
    print(f"Optimal theta: {prob.value:.2f}")
    print()
    print("Positions:")
    print(f"  {'Contract':<14} {'Qty':>8} {'Δ-contrib':>10} {'θ-contrib':>10} {'γ-contrib':>10}")
    print(f"  {'-'*14} {'-'*8} {'-'*10} {'-'*10} {'-'*10}")
    for i in range(n):
        qty = quantity.value[i]
        d_contrib = deltas[i] * qty
        t_contrib = thetas[i] * qty
        g_contrib = gammas[i] * qty
        print(f"  {labels[i]:<14} {qty:>8.2f} {d_contrib:>10.2f} {t_contrib:>10.2f} {g_contrib:>10.3f}")
    print()
    print(f"Net delta: {(deltas @ quantity.value):.4f}")
    print(f"Net theta: {(thetas @ quantity.value):.2f}")
    print(f"Net gamma: {(gammas @ quantity.value):.4f}")
    print(f"Gamma floor: {GAMMA_FLOOR}")


if __name__ == "__main__":
    main()
