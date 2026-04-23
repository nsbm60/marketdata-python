"""
Phase 0, Problem 5: Theta-weighted DTE target constraint.

Introduces duration management — the "how spread across time" dimension
of portfolio shape. This is the fixed-income-style "duration target"
concept adapted for options.

Without this, the solver biases toward short DTE (higher theta per contract).
With it, the portfolio's theta comes from a balanced mix of DTEs.

Candidates span multiple DTEs (from 7 to 60 days). The LP must hit a
target theta-weighted DTE with a tolerance range.
"""

import numpy as np
import cvxpy as cp


def main():
    # --- Candidates spanning different DTEs ---
    # Format: (underlying, label, delta, theta, gamma, price, strike, dte)
    # Short-dated options have higher gamma and higher theta per contract.
    # Longer-dated options have lower gamma and lower theta per contract.
    candidates = [
        # NVDA — 7 DTE (weekly)
        ("NVDA", "NVDA 7d ATM-C (K=200)",   0.50, -15.0, 0.12, 2.50, 200.0,  7),
        ("NVDA", "NVDA 7d ATM-P (K=200)",  -0.50, -14.0, 0.12, 2.30, 200.0,  7),
        ("NVDA", "NVDA 7d OTM-C (K=210)",   0.25,  -8.0, 0.07, 1.00, 210.0,  7),
        ("NVDA", "NVDA 7d OTM-P (K=190)",  -0.25,  -9.0, 0.07, 1.20, 190.0,  7),
        # NVDA — 21 DTE (monthly-ish)
        ("NVDA", "NVDA 21d ATM-C (K=200)",  0.50,  -8.0, 0.06, 4.20, 200.0, 21),
        ("NVDA", "NVDA 21d ATM-P (K=200)", -0.50,  -7.0, 0.06, 3.80, 200.0, 21),
        ("NVDA", "NVDA 21d OTM-C (K=210)",  0.30,  -5.0, 0.04, 1.80, 210.0, 21),
        ("NVDA", "NVDA 21d OTM-P (K=190)", -0.30,  -6.0, 0.04, 2.10, 190.0, 21),
        # NVDA — 60 DTE (longer-dated)
        ("NVDA", "NVDA 60d ATM-C (K=200)",  0.52,  -4.5, 0.03, 7.50, 200.0, 60),
        ("NVDA", "NVDA 60d ATM-P (K=200)", -0.48,  -4.0, 0.03, 6.80, 200.0, 60),
        ("NVDA", "NVDA 60d OTM-C (K=220)",  0.30,  -3.0, 0.02, 2.90, 220.0, 60),
        ("NVDA", "NVDA 60d OTM-P (K=180)", -0.30,  -3.5, 0.02, 3.20, 180.0, 60),
    ]

    underlyings = [c[0] for c in candidates]
    labels      = [c[1] for c in candidates]
    deltas      = np.array([c[2] for c in candidates])
    thetas      = np.array([c[3] for c in candidates])
    gammas      = np.array([c[4] for c in candidates])
    prices      = np.array([c[5] for c in candidates])
    strikes     = np.array([c[6] for c in candidates])
    dtes        = np.array([c[7] for c in candidates], dtype=float)
    n = len(candidates)

    # --- Starting portfolio (empty, to see clean allocation) ---
    current_qty = np.zeros(n)

    # --- Budgets and limits ---
    TARGET_DTE          = 21.0    # want theta-weighted avg DTE near 21 days
    DTE_TOLERANCE_DAYS  = 5.0     # ±5 days around target
    GAMMA_FLOOR         = -5.0
    MAX_NOTIONAL        = 500_000.0
    DELTA_TARGET        = 0
    DELTA_TOLERANCE     = 10

    transaction_cost_per_contract = np.full(n, 0.75)

    # --- Decision variable ---
    change = cp.Variable(n, name="change", integer=True)
    final_qty = current_qty + change

    # --- Derived quantities ---
    contract_values = prices * 100
    notionals_per_contract = strikes * 100

    notional_exposure = cp.sum(cp.multiply(notionals_per_contract, cp.abs(final_qty)))

    # For theta harvest, we want theta_i * quantity_i > 0 (collecting theta).
    # Since thetas are all negative, we want quantity < 0 (short) to get positive contribution.
    # Let's define "theta contribution" explicitly.
    theta_contributions = cp.multiply(thetas, final_qty)  # element-wise product
    total_theta = cp.sum(theta_contributions)

    # --- Theta-weighted DTE constraint ---
    # Ideal formula: Σ(dte_i × theta_i × qty_i) / Σ(theta_i × qty_i) = TARGET_DTE
    # Rearranged to linear form: Σ((dte_i - TARGET_DTE) × theta_i × qty_i) = 0 (for equality)
    # With tolerance: |Σ((dte_i - TARGET_DTE) × theta_i × qty_i)| ≤ TOLERANCE × total_theta
    
    dte_weighted_theta = cp.sum(cp.multiply(dtes * thetas, final_qty))
    target_weighted_theta = TARGET_DTE * total_theta
    
    # --- Objective ---
    objective = cp.Maximize(
        total_theta
        - transaction_cost_per_contract @ cp.abs(change)
    )

    # --- Constraints ---
    constraints = [
        # Delta target
        deltas @ final_qty * 100 >= (DELTA_TARGET - DELTA_TOLERANCE),
        deltas @ final_qty * 100 <= (DELTA_TARGET + DELTA_TOLERANCE),
        
        # Gamma floor
        gammas @ final_qty >= GAMMA_FLOOR,
        
        # Position bounds
        final_qty >= -10,
        final_qty <= 10,
        
        # Notional cap
        notional_exposure <= MAX_NOTIONAL,
        
        # Theta-weighted DTE target (with tolerance)
        # |Σ((dte_i - target) × theta_i × qty_i)| ≤ tolerance × |total_theta|
        # Equivalent to: target - tolerance ≤ weighted_avg_dte ≤ target + tolerance
        # We require total_theta > 0 (collecting theta, which is what we want)
        dte_weighted_theta - target_weighted_theta >= -DTE_TOLERANCE_DAYS * total_theta,
        dte_weighted_theta - target_weighted_theta <=  DTE_TOLERANCE_DAYS * total_theta,
        
        # Ensure we're actually collecting theta (positive portfolio theta)
        total_theta >= 10.0,
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.HIGHS)

    # --- Report ---
    print("=" * 115)
    print(f"Solver status: {prob.status}")
    if prob.status != "optimal":
        print("No solution found.")
        return
    
    print(f"Objective (theta - costs): {prob.value:.2f}")
    print()
    print("Positions (grouped by DTE):")
    print(f"  {'Contract':<28} {'DTE':>4} {'Price':>6} {'Qty':>6} "
          f"{'Δ-contrib':>10} {'θ-contrib':>10} {'γ-contrib':>10}")
    print(f"  {'-'*28} {'-'*4} {'-'*6} {'-'*6} {'-'*10} {'-'*10} {'-'*10}")
    for i in range(n):
        qty = final_qty.value[i]
        d_c = deltas[i] * qty
        t_c = thetas[i] * qty
        g_c = gammas[i] * qty
        print(f"  {labels[i]:<28} {int(dtes[i]):>4} {prices[i]:>6.2f} {qty:>6.1f} "
              f"{d_c:>10.2f} {t_c:>10.2f} {g_c:>10.3f}")

    # Compute theta-weighted DTE
    final_values = final_qty.value
    theta_contribs = thetas * final_values
    total_theta_val = theta_contribs.sum()
    
    if total_theta_val > 0:
        weighted_dte = (dtes * theta_contribs).sum() / total_theta_val
    else:
        weighted_dte = float('nan')
    
    # DTE distribution
    print()
    print("Theta distribution by DTE:")
    unique_dtes = sorted(set(dtes))
    for dte in unique_dtes:
        mask = dtes == dte
        dte_theta = theta_contribs[mask].sum()
        dte_pct = (dte_theta / total_theta_val * 100) if total_theta_val > 0 else 0
        print(f"  DTE {int(dte):>3}: theta ${dte_theta:>8.2f}  ({dte_pct:>5.1f}% of total)")
    
    print()
    print(f"Portfolio totals:")
    print(f"  Net delta:           {(deltas @ final_values * 100):>8.2f}  (target: {DELTA_TARGET:+d} ± {DELTA_TOLERANCE})")
    print(f"  Net gamma:           {(gammas @ final_values):>8.3f}  (floor: {GAMMA_FLOOR})")
    print(f"  Net theta:           {total_theta_val:>8.2f}")
    print(f"  Theta-weighted DTE:  {weighted_dte:>8.2f}  (target: {TARGET_DTE:.0f} ± {DTE_TOLERANCE_DAYS:.0f})")
    print(f"  Notional exposure:   ${(notionals_per_contract @ np.abs(final_values)):>10,.0f}  (limit: ${MAX_NOTIONAL:,.0f})")


if __name__ == "__main__":
    main()