# Tier-2 QuantLib conventions

Cross-check of methodology Black-76 (py_vollib) against QuantLib
`blackFormula` / `BlackCalculator`. Tolerances (plan): **IV 1e-8**, **greeks 1e-6**.

## Shared model

| Quantity | Definition |
|----------|------------|
| Year fraction | Continuous ACT/365 on calendar time (`T` in years) |
| Discount | \(D = e^{-rT}\) continuous |
| Price | Discounted Black-76 (forward form) |
| Expiry clock | Not used in Tier-2 (fixtures supply `T` directly) |

Both engines receive the same `(F, K, T, D, σ, right)`.

## Unit alignment (required)

| Greek | QuantLib raw | Methodology (py_vollib) | Conversion applied |
|-------|--------------|-------------------------|--------------------|
| delta | forward δ | spot δ when `S` given: `δ_fwd * F/S` | same transform |
| gamma | ∂²P/∂F² | same | none |
| vega | ∂P/∂σ (per 1.0 vol) | per **1 vol point** (0.01) | **QL / 100** |
| theta | ∂P/∂t annualized | **per calendar day** | **QL `thetaPerDay`** |
| rho | see below | see below | **documented split** |

## Rho — accepted systematic difference

**QuantLib `BlackCalculator.rho(T)`** is cash BS-style rho:

\[
\rho_{\mathrm{cash}}^{\mathrm{call}} = T \cdot D \cdot K \cdot N(d_2)
\]

(per unit rate; we report `/100` for “per 1%”).

**py_vollib `black` rho** (and our methodology) holds the **forward fixed** and only
differentiates the discount factor:

\[
\rho_{\mathrm{futures}} = -T \cdot \mathrm{price}
\]

(per unit rate; `/100` per 1%).

These are **different quantities**. For ATM short-dated options they diverge by
O(0.01–0.25) in 1%-units. Tier-2 therefore:

- Compares **delta, gamma, vega, theta** to py_vollib at **1e-6**
- Compares **`rho_futures`** (F-fixed closed form from QL price) to py_vollib rho at **1e-6**
- Does **not** gate on QuantLib cash rho vs py_vollib (documented exception)

## IV recovery

`blackFormulaImpliedStdDev` needs a non-default **stdDev guess** at tiny `T`
(sub-day fixtures). We seed `guess = 0.25 * sqrt(T)` and `accuracy=1e-14`.
With that, recovered IV vs fixture `true_vol` is well inside **1e-8** relative
across the Tier-1 grid.

## Price parity

`blackFormula` vs py_vollib `black` on the Tier-1 grid: max |Δprice| ~ 1e-14
(floating noise only). No accepted price bias.

## Install

```bash
pip install -e ".[tier2]"
# or
pip install QuantLib
```

Tests use `pytest.importorskip("QuantLib")` so Tier-1 CI stays green without QL.
