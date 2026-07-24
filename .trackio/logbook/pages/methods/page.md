# Methods


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_methods_v2", "created_at": "2026-07-25T00:00:00+00:00", "title": "Protocol and limitations"}
-->
# Methods

**Ridge regression (Claims 1–4):** Exact full-batch GD via eigendecomposition of
XXᵀ. Null-space coordinates decay by (1−ηλ)^t. Population loss computed in
closed form (no Monte Carlo). 7 numerical tests verify spectral state matches
explicit GD, population formula matches Monte Carlo, integer threshold
semantics, Eq.(8) bounds, deterministic generation, arbitrary teacher types,
and ReLU grokking.

**Random features (Claim 5, Figure 3):** Fixed ReLU feature map with w_j ~
N(0,ν²/d I_d), features normalized by 1/√m. Realizable teacher (random θ* in
feature space). Spectral GD on output-layer weights only.

**Two-layer ReLU (Claim 5, Figure 4):** Full-batch GD on both layers with zero
teacher. η=0.01 (deviation from paper's 10⁻⁴). Gradient clipping for stability.

**Scope:** All 5 scored claims. Ridge claims use exact spectral methods. ReLU
experiments use documented deviations for CPU feasibility. The Gaussian feature
distribution is unbounded; this is the paper's finite-sample specialization.
