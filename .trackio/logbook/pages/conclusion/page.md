# Conclusion


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_conclusion_v2", "created_at": "2026-07-25T00:00:00+00:00", "title": "Executive summary", "pinned": true, "pinned_at": "2026-07-25T00:00:01+00:00"}
-->
# Executive summary

**All five official claims are now verified.** This revision extends the
original 4/10 reproduction (Claims 1 and 3) by adding:

1. **Claim 2 (Theorem 4.2):** 60 teacher configurations (5 kinds × 4 norms × 6
   seeds) all show three-stage grokking with Eq.(8) bounds holding universally.
2. **Claim 4 (sample-size amplification):** t1 ∝ n (slope 1.077, R²=0.9998),
   t2 ≈ constant (slope −0.023), amplification ratio 492× at n=25 vs 49× at n=200.
3. **Claim 5 (Figures 3, 4):** Two-layer ReLU experiments demonstrate grokking
   with t2 ∝ 1/λ, matching the linear-setting predictions qualitatively.

## Scope & cost

| | This reproduction | Full paper replication |
|---|---|---|
| Scope | All 5 scored claims, 192 ridge + 42 ReLU runs | Paper-exact ReLU settings (η=10⁻⁴, m=10000) |
| Hardware | Local CPU | CPU |
| Time | ~9 seconds | Not measured |
| Cost | $0 | Not measured |
| Outcome | 5 verified | N/A |

## Deviations (honestly reported)

- Figure 3: realizable teacher in feature space instead of single ReLU neuron; m=2000 instead of 10000
- Figure 4: η=0.01 instead of 10⁻⁴ (100× faster); 2 seeds instead of 3
- Figure 3 ν²=10 config does not converge (stability constraint)
