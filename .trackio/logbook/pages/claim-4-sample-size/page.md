# Claim 4 - sample size


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_c4_ss", "created_at": "2026-07-25T00:00:00+00:00", "title": "Sample-size amplification"}
-->
# Claim 4 — VERIFIED

**Figure 2 (right upper):** "decreasing sample size amplifies grokking by
speeding up the convergence of the training loss (affects t1)."

## Protocol

Sample-size sweep: n ∈ {25, 50, 100, 200} with m=1000, η=1, λ=10⁻⁴, ν²=1,
6 seeds per config, ε=c=0.01.

## Results

| n | Mean t1 | Mean t2 | t2/(t1+1) median | Eq.(8) t1 passes | Eq.(8) t2 passes |
|---|---|---|---|---|---|
| 25 | 47.7 | 23,376 | 492× | 6/6 | 6/6 |
| 50 | 103.5 | 23,184 | 220× | 6/6 | 6/6 |
| 100 | 214.8 | 22,857 | 108× | 6/6 | 6/6 |
| 200 | 449.7 | 22,260 | 49× | 6/6 | 6/6 |

**Quantitative verification:**
- t1 vs n log-log slope: **1.077** (R²=0.9998) — confirms t1 ∝ n
- t2 vs n log-log slope: **−0.023** (R²=0.94) — confirms t2 ≈ independent of n
- Grokking amplification: t2/(t1+1) ratio increases **10×** from n=200 to n=25

The prediction from Eq.(8): t1 ≤ n·ln(14mν²/ε) / (2η·λ⁺_min(ΦᵀΦ)) — the
linear dependence on n is confirmed experimentally.

## Weight decay amplification (previously verified)

Across λ=2⁻¹⁹…2⁻¹⁵, the t2 log-log slope is −1.000003 with R²≈1, and delay
changes by 16.04×. Combined with the sample-size result, both halves of
Figure 2's amplification claim are now verified.

Data: `outputs/full/runs.csv` (rows with `sweep=sample_size`)
Git SHA: b0435b8
