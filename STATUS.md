# STATUS — Provable Grokking in Ridge Regression

**OpenReview:** `5nNNVY8NW4` · **State:** published — 5/5 · **Updated:** 2026-07-25

Full reproduction with 192 ridge + 42 ReLU runs in ~9 seconds. All five claims
verified: three-stage grokking (6/6 seeds), arbitrary teachers (60/60 configs),
Eq.(8) bounds (126/126 rows), sample-size amplification (t1∝n slope 1.077),
and two-layer ReLU grokking (Fig 3: 36/42, Fig 4: 10/10). Seven tests pass.

**Space:** https://huggingface.co/spaces/DineshAI/5nNNVY8NW4
**Published Space SHA:** `510487bdb340042500b03c93cf65caf29c3a427e`
**GitHub:** https://github.com/MachineLearning-Nerd/icml26-repro-5nNNVY8NW4-grokking
