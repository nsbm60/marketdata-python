"""
Phase 0, Problem 1: Simplest LP with realistic Greeks.

5 fictional option contracts. All options have negative theta (real physics).
The solver chooses whether to go long (qty > 0, pays theta) or short
(qty < 0, collects theta).

Objective: maximize portfolio theta contribution.
Constraint: net delta = 0, quantities bounded to [-10, +10].

Expected: solver shorts options to collect theta, balanced for delta = 0.
"""

import numpy as np
import cvxpy as cp


def main():
    # Five option contracts — all have negative theta (real physics).
    # Index | delta  | theta  | Rough interpretation
    #   0   | +0.50  | -10    | ATM call
    #   1   | -0.50  |  -8    | ATM put
    #   2   | +0.30  |  -5    | OTM call (closer to OTM, less gamma, smaller theta)
    #   3   | -0.30  |  -6    | OTM put
    #   4   | +0.10  |  -2    | Far-OTM call

    deltas = np.array([0.50, -0.50, 0.30, -0.30, 0.10])
    thetas = np.array([-10.0, -8.0, -5.0, -6.0, -2.0])
    n = len(deltas)

    labels = [
        "ATM-Call",
        "ATM-Put",
        "OTM-Call",
        "OTM-Put",
        "Far-OTM-Call",
    ]

    # Decision variable: quantities (positive = long, negative = short)
    quantity = cp.Variable(n, name="quantity")

    # Objective: maximize theta contribution = sum(theta_i * quantity_i)
    objective = cp.Maximize(thetas @ quantity)

    constraints = [
        deltas @ quantity == 0,
        quantity >= -10,
        quantity <=  10,
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.HIGHS)

    print("=" * 60)
    print(f"Solver status: {prob.status}")
    print(f"Optimal theta (objective value): {prob.value:.2f}")
    print()
    print("Positions:")
    print(f"  {'Contract':<14} {'Qty':>8} {'Delta-contrib':>15} {'Theta-contrib':>15}")
    print(f"  {'-'*14} {'-'*8} {'-'*15} {'-'*15}")
    for i in range(n):
        qty = quantity.value[i]
        d_contrib = deltas[i] * qty
        t_contrib = thetas[i] * qty
        print(f"  {labels[i]:<14} {qty:>8.2f} {d_contrib:>15.2f} {t_contrib:>15.2f}")
    print()
    print(f"Net delta: {(deltas @ quantity.value):.4f}")
    print(f"Net theta: {(thetas @ quantity.value):.2f}")


if __name__ == "__main__":
    main()
