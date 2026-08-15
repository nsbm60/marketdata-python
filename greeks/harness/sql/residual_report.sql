-- ClickHouse residual acceptance queries (PR6)
-- Parameter: methodology_version
-- System of record tables: trading.greeks_residuals, trading.greeks_validation

-- Gate: NTM 0.95–1.05, 3–30 DTE, matched only
SELECT
    count() AS n,
    median(abs(residual_iv_bps)) AS median_abs_iv_bps,
    median(abs(residual_delta)) AS median_abs_delta,
    (median(abs(residual_iv_bps)) <= 20)
        AND (median(abs(residual_delta)) <= 0.005) AS gate_pass
FROM trading.greeks_residuals
WHERE methodology_version = {methodology_version:String}
  AND join_class = 'matched'
  AND moneyness BETWEEN 0.95 AND 1.05
  AND (dte_years * 365) BETWEEN 3 AND 30
  AND residual_iv_bps IS NOT NULL
  AND residual_delta IS NOT NULL;

-- All buckets (moneyness × DTE)
SELECT
    multiIf(
        dte_years * 365 * 24 < 2, '<2h',
        dte_years * 365 * 24 < 6.5, '2h-6.5h',
        dte_years * 365 < 1, '6.5h-1d',
        dte_years * 365 < 3, '1d-3d',
        dte_years * 365 < 7, '3d-7d',
        dte_years * 365 < 30, '7d-30d',
        dte_years * 365 <= 90, '30d-90d',
        '>90d'
    ) AS dte_bucket,
    multiIf(
        moneyness < 0.85, 'mny<0.85',
        moneyness < 0.95, '0.85-0.95',
        moneyness <= 1.05, '0.95-1.05',
        moneyness <= 1.15, '1.05-1.15',
        'mny>1.15'
    ) AS mny_bucket,
    count() AS n,
    median(abs(residual_iv_bps)) AS median_abs_iv_bps,
    median(abs(residual_delta)) AS median_abs_delta
FROM trading.greeks_residuals
WHERE methodology_version = {methodology_version:String}
  AND join_class = 'matched'
  AND residual_iv_bps IS NOT NULL
  AND residual_delta IS NOT NULL
GROUP BY dte_bucket, mny_bucket
ORDER BY mny_bucket, dte_bucket;

-- Validation accounting (AC5)
SELECT
    status,
    count() AS n
FROM trading.greeks_validation
WHERE methodology_version = {methodology_version:String}
GROUP BY status;
