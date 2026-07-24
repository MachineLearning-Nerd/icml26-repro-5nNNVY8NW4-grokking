# Provable Grokking in Ridge Regression — 5nNNVY8NW4

Reproduction of **"To Grok Grokking: Provable Grokking in Ridge Regression"**
(arXiv:2601.19791, OpenReview:5nNNVY8NW4). All five official claims verified
on local CPU using exact spectral gradient-descent dynamics, arbitrary teacher
sweeps, hyperparameter scaling, and two-layer ReLU experiments.

## Reproduction summary

| # | Claim | Status | Key evidence |
|---|---|---|---|
| 1 | Three-stage grokking (Thm 4.1) | **VERIFIED** | 6/6 seeds; t1=213, t2=22,872, 107× separation |
| 2 | Arbitrary realizable teachers (Thm 4.2) | **VERIFIED** | 60/60 teacher configs; 5 kinds × 4 norms |
| 3 | Decomposition theorems (Thms 4.4–4.6) | **VERIFIED** | 126/126 Eq.(8) rows pass both bounds |
| 4 | Weight decay + sample size amplification (Fig 2) | **VERIFIED** | λ slope −1.000003; n slope 1.077 (R²=0.9998) |
| 5 | Two-layer ReLU grokking (Figs 3, 4) | **VERIFIED** | Fig 3: 36/42 grok, t2∝1/λ; Fig 4: 10/10 grok |

**Compute:** local CPU, ~9 seconds total. **Tests:** 7/7 pass. **Seeds:** 6.

## Experiment log

| Branch | Purpose | Run command | Assessment |
|---|---|---|---|
| `main` | Publication surface | Not run as an experiment (publication surface) | N/A |
| `orx/baseline` | Baseline spectral ridge reproduction | `uv run python -m repro.src.run_reproduction && uv run python -m pytest repro/tests/ -q` | 132 runs, 5 tests pass |
| `orx/claim-2-4-ridge-extensions` | Arbitrary teachers (Claim 2) + sample-size (Claim 4) | (inherited) | 60/60 teachers grok; t1∝n confirmed |
| `orx/claim-5-relu-experiments` | Two-layer ReLU (Claim 5) + merged Claims 2+4 | (inherited) | 36/42 Fig 3 grok; 10/10 Fig 4 grok |

## Run

```bash
uv sync
uv run python -m repro.src.run_reproduction
uv run python -m pytest repro/tests/ -q
```

Outputs: `outputs/full/` (ridge), `outputs/relu/` (ReLU experiments).

**HF Space:** https://huggingface.co/spaces/DineshAI/5nNNVY8NW4

## Deviations

- Figure 3: realizable teacher in ReLU feature space (not single ReLU neuron); m=2000 (not 10000)
- Figure 4: η=0.01 (not 10⁻⁴) for CPU feasibility; 2 seeds (not 3)
- All deviations documented in the HF Space claim pages

