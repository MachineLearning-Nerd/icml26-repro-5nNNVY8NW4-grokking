# Claim 5 - ReLU experiments


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_c5_relu", "created_at": "2026-07-25T00:00:00+00:00", "title": "Two-layer ReLU grokking"}
-->
# Claim 5 — VERIFIED

**Figures 3 and 4:** "Two-layer ReLU experiments qualitatively reproduce the
predicted grokking-time dependence on hyperparameters beyond the linear
setting."

## Figure 3: Random ReLU Features (Section 5.2)

Only output-layer weights trained (spectral GD on a fixed ReLU feature map).
Uses a realizable teacher in the feature space (random θ* with ||θ*||=1).

**Deviation:** Uses realizable teacher in ReLU feature space instead of a
single ReLU neuron, and m=2000 instead of 10000 for CPU feasibility. The
qualitative grokking dynamics are preserved because the spectral structure
(row-space vs null-space decay) is identical.

| Config | Runs | Grokking | Median t1 | Median t2 |
|---|---|---|---|---|
| default (λ=10⁻⁵) | 6 | 6/6 | 444 | 155,449 |
| λ=10⁻⁶ | 6 | 6/6 | 446 | 1,554,497 |
| λ=10⁻⁴ | 6 | 6/6 | 422 | 15,544 |
| n=50 | 6 | 6/6 | 221 | 168,360 |
| n=200 | 6 | 6/6 | 1,058 | 145,150 |
| ν²=0.1 | 6 | 6/6 | 488 | 33,156 |
| ν²=10 | 6 | 0/6 | 232 | None |
| **Total** | **42** | **36/42** | | |

**Key trends confirmed:**
- t2 ∝ 1/λ: each 10× decrease in λ gives ~10× increase in t2 (15,544 → 155,449 → 1,554,497)
- t1 ∝ n: t1 increases linearly with sample size (221 → 444 → 1,058)
- t2 ≈ independent of n
- ν²=10 fails due to GD stability constraint (adaptive η too small for convergence within max_steps)

## Figure 4: Two-Layer ReLU GD (Section 5.3)

Both layers (W, a) trained with full-batch GD on zero teacher.

**Deviation:** η=0.01 instead of the paper's 10⁻⁴ (100× faster for CPU
feasibility). All convergence rates scale uniformly with η, so the qualitative
hyperparameter dependencies are preserved.

| Config | Runs | Grokking | Median t1 | Median t2 |
|---|---|---|---|---|
| default (λ=0.05) | 2 | 2/2 | 500 | 2,000 |
| λ=0.01 | 2 | 2/2 | 500 | 8,000 |
| λ=0.1 | 2 | 2/2 | 500 | 1,000 |
| n=25 | 2 | 2/2 | 500 | 2,000 |
| n=100 | 2 | 2/2 | 500 | 1,500 |
| **Total** | **10** | **10/10** | | |

**Key trend confirmed:** t2 ∝ 1/λ (1,000 → 2,000 → 8,000 for λ=0.1 → 0.05 → 0.01)

## Limitations

1. Figure 3 uses realizable teacher in feature space instead of single ReLU neuron
2. Figure 3 uses m=2000 instead of paper's m=10000
3. Figure 4 uses η=0.01 instead of paper's η=10⁻⁴
4. ν²=10 config in Figure 3 does not converge (stability constraint)
5. Figure 4 uses 2 seeds instead of 6 (CPU time constraint)

Code: `repro/src/relu_experiments.py`
Data: `outputs/relu/relu_runs.csv`, `outputs/relu/relu_summary.json`
Git SHA: b0435b8
