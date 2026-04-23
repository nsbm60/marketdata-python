"""
Phase 0, Problem 3: Existing positions with gross and net cash flow constraints.

Builds on Problem 2. Same options, same Greeks. Now with:
  - Existing positions (starting portfolio, not empty)
  - Decision variable = CHANGE from current quantities
  - Per-contract transaction costs (commission + half-spread)
  - Gross premium turnover budget (operational activity cap)
  - Net cash outflow cap (don't let rebalances cost too much to execute)
  - Notional exposure cap based on strikes
"""

import numpy as np
import cvxpy as cp


def main():
    # --- Option contracts with realistic Greeks and prices ---
    deltas = np.array([0.50, -0.50, 0.30, -0.30, 0.10])
    thetas = np.array([-10.0, -8.0, -5.0, -6.0, -2.0])
    gammas = np.array([0.08, 0.08, 0.05, 0.05, 0.02])
    prices = np.array([3.50, 2.80, 1.25, 1.65, 0.30])  # mid price per share
    strikes = np.array([200.0, 200.0, 210.0, 190.0, 225.0])
    n = len(deltas)

    labels = [
        "ATM-Call (K=200)",
        "ATM-Put  (K=200)",
        "OTM-Call (K=210)",
        "OTM-Put  (K=190)",
        "Far-OTM-Call (K=225)",
    ]

    # --- Per-contract transaction costs (commission + half-spread) ---
    transaction_cost_per_contract = np.array([1.00, 1.00, 0.75, 0.75, 0.50])

    # --- Existing portfolio ---
    # Negative quantity = short position.
    current_qty = np.array([-8.0, -8.0, 0.0, -5.0, 0.0])

    # --- Budgets and limits ---
    # Gross premium turnover: caps total dollar activity regardless of direction.
    # Every buy or sell counts. Reflects operational capacity (commissions,
    # attention, execution risk).
    MAX_GROSS_TURNOVER = 4000.0

    # Net cash outflow: caps how much CASH can flow OUT of our account net
    # of cash coming in. Positive = we paid to rebalance. Negative = we received
    # cash from rebalancing. For a premium harvester, this is typically negative
    # (net inflow). We want to cap the UPSIDE: never pay more than this.
    # Setting to 200 allows small net debits for necessary repositioning.
    # Setting to 0 would require cash-neutral or credit rebalances.
    MAX_CASH_OUTFLOW = 200.0

    # Total notional exposure of final portfolio (strike-based).
    MAX_NOTIONAL_EXPOSURE = 500000.0

    # Gamma floor: prevents excessive short gamma.
    GAMMA_FLOOR = -2.0

    # --- Decision variable: change from current position ---
    # change > 0 means going longer (buying contracts, paying premium)
    # change < 0 means going shorter (selling contracts, receiving premium)
    change = cp.Variable(n, name="change")
    final_qty = current_qty + change

    # --- Derived quantities ---
    contract_values = prices * 100                 # premium per contract in dollars
    notionals_per_contract = strikes * 100         # notional per contract (strike-based)

    # Gross premium turnover: absolute dollar activity, both directions
    gross_turnover = cp.sum(cp.multiply(contract_values, cp.abs(change)))

    # Net cash outflow: signed dollar flow
    #   change > 0 (buying) contributes positive value = cash paid
    #   change < 0 (selling) contributes negative value = cash received
    # Sum is total net cash OUTFLOW from our account.
    net_cash_outflow = cp.sum(cp.multiply(contract_values, change))

    # Notional exposure of final portfolio (absolute = total risk in play)
    notional_exposure = cp.sum(cp.multiply(notionals_per_contract, cp.abs(final_qty)))

    # --- Objective: maximize theta minus per-contract transaction costs ---
    objective = cp.Maximize(
        thetas @ final_qty
        - transaction_cost_per_contract @ cp.abs(change)
    )

    # --- Constraints ---
    constraints = [
        deltas @ final_qty == 0,                        # delta neutral
        gammas @ final_qty >= GAMMA_FLOOR,              # gamma floor
        final_qty >= -10,                               # per-position lower bound
        final_qty <= 10,                                # per-position upper bound
        gross_turnover <= MAX_GROSS_TURNOVER,           # operational activity cap
        net_cash_outflow <= MAX_CASH_OUTFLOW,           # don't pay too much to rebalance
        notional_exposure <= MAX_NOTIONAL_EXPOSURE,     # portfolio notional cap
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.HIGHS)

    # --- Report ---
    print("=" * 100)
    print(f"Solver status: {prob.status}")
    print(f"Objective value (theta - transaction costs): {prob.value:.2f}")
    print()
    print("Existing -> Final positions:")
    print(f"  {'Contract':<22} {'Price':>6} {'Strike':>7} {'Current':>8} {'Change':>8} "
          f"{'Final':>8} {'Notional':>10} {'θ-contrib':>10}")
    print(f"  {'-'*22} {'-'*6} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*10}")
    for i in range(n):
        cur = current_qty[i]
        chg = change.value[i]
        fin = final_qty.value[i]
        notional = abs(fin) * notionals_per_contract[i]
        t_c = thetas[i] * fin
        print(f"  {labels[i]:<22} {prices[i]:>6.2f} {strikes[i]:>7.2f} "
              f"{cur:>8.2f} {chg:>+8.2f} {fin:>8.2f} "
              f"{notional:>10.2f} {t_c:>10.2f}")

    # --- Summary metrics ---
    total_gross_turnover = np.sum(np.abs(change.value) * contract_values)
    total_net_cash_outflow = np.sum(change.value * contract_values)
    total_transaction_cost = np.sum(np.abs(change.value) * transaction_cost_per_contract)
    total_notional = np.sum(np.abs(final_qty.value) * notionals_per_contract)
    net_theta = thetas @ final_qty.value

    print()
    print(f"Net delta:              {(deltas @ final_qty.value):.4f}")
    print(f"Net gamma:              {(gammas @ final_qty.value):.4f}  (floor: {GAMMA_FLOOR})")
    print(f"Net theta:              {net_theta:.2f}")
    print(f"Gross turnover:         ${total_gross_turnover:.2f}  (budget: ${MAX_GROSS_TURNOVER:.0f})")
    print(f"Net cash outflow:       ${total_net_cash_outflow:+.2f}  (cap: ${MAX_CASH_OUTFLOW:+.0f})")
    if total_net_cash_outflow < 0:
        print(f"  (negative = cash INFLOW of ${-total_net_cash_outflow:.2f})")
    print(f"Notional exposure:      ${total_notional:.2f}  (limit: ${MAX_NOTIONAL_EXPOSURE:.0f})")
    print(f"Transaction costs:      ${total_transaction_cost:.2f}")
    print(f"Net theta after costs:  {net_theta - total_transaction_cost:.2f}")


if __name__ == "__main__":
    main()