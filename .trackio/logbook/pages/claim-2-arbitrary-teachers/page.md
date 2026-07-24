# Claim 2 - arbitrary teachers


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_c2_at", "created_at": "2026-07-25T00:00:00+00:00", "title": "Arbitrary teacher verification"}
-->
# Claim 2 — VERIFIED

**Theorem 4.2** states that for *any* realizable teacher θ*, under appropriate
hyperparameter conditions, grokking occurs with quantitative bounds on t1 and t2.

## Protocol

We test 10 structurally distinct teacher configurations × 6 seeds = 60 runs at
the paper-default Gaussian setting (n=100, m=1000, η=1, λ=10⁻⁴, ν²=1, ε=c=0.01).

Teacher types tested:
- **random** with norms 0.1, 0.5, 1.0, 2.0 (tests norm universality)
- **sparse** with sparsity k=5, 10, 50 (tests structural diversity)
- **one_hot** (single coordinate, tests extreme sparsity)
- **uniform** (all entries equal, tests dense uniform structure)
- **top_k** (first k entries, tests aligned structure)

## Results

| Teacher kind | Runs | Grokking | Eq.(8) t1 bound | Eq.(8) t2 bound | Median t1 | Median t2 |
|---|---|---|---|---|---|---|
| random | 24 | 24/24 | 24/24 | 24/24 | 213 | 22,758 |
| sparse | 18 | 18/18 | 18/18 | 18/18 | 213 | 22,936 |
| one_hot | 6 | 6/6 | 6/6 | 6/6 | 213 | 23,008 |
| uniform | 6 | 6/6 | 6/6 | 6/6 | 213 | 22,932 |
| top_k | 6 | 6/6 | 6/6 | 6/6 | 213 | 22,840 |
| **Total** | **60** | **60/60** | **60/60** | **60/60** | | |

**Key finding:** t1 and t2 are essentially invariant across teacher types (t1
range: 202–230, t2 range: 22,160–24,995), confirming that the Eq.(8) bounds
(which do not depend on θ*) hold universally. The three-stage grokking dynamics
are identical regardless of teacher structure.

## Implementation

Code: `repro/src/ridge_dynamics.py` (`_make_teacher` method, `teacher_kind` parameter)
Data: `outputs/full/runs.csv` (rows with `sweep=teacher_type`)
Git SHA: b0435b8 (branch `orx/claim-5-relu-experiments`)

Run command: `uv run python -m repro.src.run_reproduction && uv run python -m pytest repro/tests/ -q`
