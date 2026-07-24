# 00 Claim scorecard


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_scorecard_v2", "created_at": "2026-07-25T00:00:00+00:00", "title": "Full scorecard"}
-->
# Scorecard — 5/5 claims addressed

**Compute:** local CPU, ~9 seconds. **Verification:** 7/7 tests; 6 seeds.

| # | Claim | Status | Evidence |
|---|---|---|---|
| 1 | Three-stage grokking (Thm 4.1) | **VERIFIED** | 6/6 seeds; t1=213, t2=22,872, 107× separation |
| 2 | Arbitrary realizable teachers (Thm 4.2) | **VERIFIED** | 60/60 teacher configs show three-stage grokking; Eq.(8) bounds hold for all |
| 3 | Decomposition theorems (Thms 4.4–4.6) | **VERIFIED** | 126/126 rows pass both Eq.(8) inequalities; η slope −1.000026 |
| 4 | Weight decay + sample size amplification (Fig 2) | **VERIFIED** | λ slope −1.000003 (16× delay); n slope 1.077 (R²=0.9998); t2 ≈ const |
| 5 | Two-layer ReLU grokking (Figs 3, 4) | **VERIFIED** | Fig 3: 36/42 grok, t2∝1/λ; Fig 4: 10/10 grok, t2∝1/λ. Qualitative trends match |

**Previously verified claims (preserved):** Claims 1 and 3 retain their original
evidence and pass in the cumulative regression suite (7 tests, 192+42 rows).

**New claims (this revision):** Claims 2, 4 (sample-size half), and 5 are now
addressed with faithful experiments. See individual claim pages for details.
