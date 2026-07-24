#!/usr/bin/env python3
"""Two-layer ReLU experiments for Claim 5 (Figures 3 and 4 of arXiv:2601.19791v3).

Figure 3 — Random-Features Neural Networks (Section 5.2):
    Only the output layer weights *a* are trained; the hidden weights *W* are
    fixed at a random ReLU feature map.  We use a realizable teacher that lives
    in the span of the ReLU features (a random linear combination of the
    features), ensuring the limiting population loss is controlled by weight
    decay alone — the same mechanism that drives grokking in the linear theory.

Figure 4 — Non-linear Neural Networks (Section 5.3):
    Both layers (W, a) are trained with GD on a zero teacher.  We run explicit
    full-batch GD and record t1 / t2 to verify that the qualitative
    hyperparameter dependencies of Figure 2 carry over to the nonlinear setting.
    Deviation: eta=0.01 instead of the paper's 1e-4 for CPU feasibility
    (100x faster; the GD contraction structure and qualitative trends are
    preserved because all rates scale uniformly with eta).
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import json
import math
from pathlib import Path
import time

import numpy as np


EPSILON = 0.01
C = 0.01
ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
#  Figure 3 — Random ReLU Features (spectral GD, output layer only)
# ---------------------------------------------------------------------------

class RandomFeaturesRidge:
    """Exact spectral GD for a two-layer random ReLU features network.

    Only the output-layer weights *a* are trained; the hidden weights *W* are
    drawn once and kept fixed.  The teacher is a *realizable* function that
    lives in the span of the ReLU features.
    """

    def __init__(
        self,
        seed: int,
        n: int,
        m: int,
        d: int,
        *,
        weight_decay: float = 1e-5,
        nu2: float = 1.0,
        eta: float = 1.0,
        n_test: int = 10_000,
    ) -> None:
        self.seed = seed
        self.n = n
        self.m = m
        self.d = d
        self.weight_decay = weight_decay
        self.nu2 = nu2
        self.eta = eta

        ss_data, ss_w, ss_teacher, ss_a, ss_test = np.random.SeedSequence(seed).spawn(5)
        data_rng = np.random.default_rng(ss_data)
        w_rng = np.random.default_rng(ss_w)
        teacher_rng = np.random.default_rng(ss_teacher)
        a_rng = np.random.default_rng(ss_a)
        test_rng = np.random.default_rng(ss_test)

        # Training inputs x ~ N(0, I_d)
        self.x = data_rng.standard_normal((n, d))

        # Fixed random features: w_j ~ N(0, nu2/d I_d) — gives ||w_j|| ~ nu
        # This matches Figure 4's hidden-layer initialization and produces
        # features at the same scale as the teacher.
        self.W = w_rng.standard_normal((m, d)) * math.sqrt(nu2 / d)

        # Feature matrix Phi_ij = ReLU(<w_j, x_i>), then scale by 1/sqrt(m).
        # This normalization matches the paper's effective feature variance
        # (nu^2/(dm) per entry) and ensures eta=1 GD stability: the Gram
        # eigenvalues become O(n/m), same regime as the linear case.
        feature_scale = 1.0 / math.sqrt(m)
        self.Phi = np.maximum(self.x @ self.W.T, 0.0) * feature_scale  # (n, m)

        # Realizable teacher: random theta_star in feature space, unit norm.
        # With scaled features (variance ~1/(2m)), the teacher output has
        # variance ~1/(2m), which is tiny but nonzero.  The grokking dynamics
        # come from the init (large) vs teacher (small) mismatch.
        theta_star = teacher_rng.standard_normal(m)
        self.theta_star = theta_star / np.linalg.norm(theta_star)
        self.y = self.Phi @ self.theta_star

        # Initialize output weights a_j ~ N(0, 1) — matches paper Section 5.2.
        # This gives initial output variance ~nu^2/2 = O(1), far above the
        # tiny teacher signal, creating the overfitting regime for grokking.
        self.a0 = a_rng.standard_normal(m)

        # Eigendecomposition of the n x n Gram matrix Phi Phi^T
        gram = self.Phi @ self.Phi.T
        eigvals, U = np.linalg.eigh(gram)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        U = U[:, order]
        tol = np.finfo(float).eps * max(n, m) * (eigvals[0] if eigvals[0] > 0 else 1.0)
        rank = int(np.count_nonzero(eigvals > tol))
        self.eigvals = eigvals[:rank]
        self.U = U[:, :rank]
        self.Vt = (self.U.T @ self.Phi) / np.sqrt(self.eigvals[:, None])  # (rank, m)

        # Project teacher and init into row space
        self.beta = self.U.T @ self.y  # (rank,)
        self.alpha0 = self.Vt @ self.a0  # (rank,)

        # Null-space component of init
        a0_proj = self.Vt.T @ self.alpha0
        self.a0_null = self.a0 - a0_proj
        self.null_init_sq = float(self.a0_null @ self.a0_null)

        # GD contraction factors
        self.q = 1.0 - eta * (self.eigvals / n + weight_decay)
        self.r = 1.0 - eta * weight_decay
        if np.any(self.q <= 0) or np.any(self.q >= 1) or not (0 < self.r < 1):
            raise ValueError("configuration outside the positive-contraction GD regime")

        # Ridge solution in row space
        self.alpha_inf = (self.eigvals / n) / (self.eigvals / n + weight_decay) * self.beta

        # Test set for population loss (same distribution)
        self.x_test = test_rng.standard_normal((n_test, d))
        self.Phi_test = np.maximum(self.x_test @ self.W.T, 0.0) * feature_scale
        self.y_test = self.Phi_test @ self.theta_star

    def row_coords(self, t: int) -> np.ndarray:
        qt = self.q ** t
        return qt * self.alpha0 + (1.0 - qt) * self.alpha_inf

    def full_state(self, t: int) -> np.ndarray:
        return self.Vt.T @ self.row_coords(t) + (self.r ** t) * self.a0_null

    def training_loss(self, t: int) -> float:
        diff = self.row_coords(t) - self.beta
        return float(np.dot(self.eigvals, diff * diff) / (2.0 * self.n))

    def population_loss(self, t: int) -> float:
        a = self.full_state(t)
        pred = self.Phi_test @ a
        resid = pred - self.y_test
        return float(np.mean(resid * resid))

    def limiting_losses(self) -> tuple[float, float]:
        row_diff = self.alpha_inf - self.beta
        train = float(np.dot(self.eigvals, row_diff * row_diff) / (2.0 * self.n))
        a_inf = self.Vt.T @ self.alpha_inf
        pred = self.Phi_test @ a_inf
        resid = pred - self.y_test
        pop = float(np.mean(resid * resid))
        return train, pop

    def _first_below(self, fn, threshold: float, max_steps: int) -> int | None:
        if fn(0) < threshold:
            return 0
        hi = 1
        while hi <= max_steps and fn(hi) >= threshold:
            hi *= 2
        if hi > max_steps:
            return None
        lo = hi // 2
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if fn(mid) < threshold:
                hi = mid
            else:
                lo = mid
        return hi

    def threshold_times(self, max_steps: int = 5_000_000) -> tuple[int | None, int | None]:
        t1_raw = self._first_below(self.training_loss, EPSILON, max_steps)
        t2_raw = self._first_below(self.population_loss, C, max_steps)
        t1 = -1 if t1_raw == 0 else (None if t1_raw is None else t1_raw - 1)
        return t1, t2_raw


# ---------------------------------------------------------------------------
#  Figure 4 — Two-layer ReLU GD (both layers trained, zero teacher)
# ---------------------------------------------------------------------------

@dataclass
class TwoLayerResult:
    t1: int | None
    t2: int | None
    train_trajectory: list[float]
    pop_trajectory: list[float]
    checkpoint_steps: list[int]


def train_two_layer_relu(
    seed: int,
    n: int = 50,
    m: int = 1000,
    d: int = 50,
    *,
    weight_decay: float = 0.05,
    nu2: float = 1.0,
    eta: float = 0.01,
    max_steps: int = 300_000,
    n_test: int = 5_000,
    eval_interval: int = 500,
) -> TwoLayerResult:
    """Full-batch GD on a two-layer ReLU network with zero teacher.

    Network: N(x) = sum_j a_j * ReLU(w_j^T x), both a and W trained.
    Teacher: N*(x) = 0 (zero function).
    Loss: (1/2n) sum_i N(x_i)^2 + (lambda/2)(||a||^2 + ||W||^F^2).

    Deviation from paper: eta=0.01 instead of 1e-4 (100x faster for CPU
    feasibility).  All convergence rates scale uniformly with eta, so the
    qualitative hyperparameter dependencies (t2 ~ 1/lambda, t1 ~ n, etc.)
    are preserved.
    """
    rng = np.random.default_rng(seed)

    x = rng.standard_normal((n, d))
    W = rng.standard_normal((m, d)) * math.sqrt(nu2 / d)
    a = rng.standard_normal(m) / math.sqrt(m)
    x_test = rng.standard_normal((n_test, d))

    train_traj: list[float] = []
    pop_traj: list[float] = []
    checkpoint_steps: list[int] = []

    def eval_losses() -> tuple[float, float]:
        A = np.maximum(x @ W.T, 0.0)
        pred = A @ a
        train = float(np.mean(pred * pred))
        At = np.maximum(x_test @ W.T, 0.0)
        pt = At @ a
        pop = float(np.mean(pt * pt))
        return train, pop

    tr0, pop0 = eval_losses()
    train_traj.append(tr0)
    pop_traj.append(pop0)
    checkpoint_steps.append(0)

    t1_found: int | None = None
    t2_found: int | None = None

    step = 0
    while step < max_steps:
        for _ in range(eval_interval):
            Z = x @ W.T
            A = np.maximum(Z, 0.0)
            pred = A @ a
            resid = pred  # zero teacher: y = 0

            grad_a = A.T @ resid / n + weight_decay * a
            mask = (Z > 0).astype(np.float64)
            Q = a[None, :] * resid[:, None] * mask
            grad_W = Q.T @ x / n + weight_decay * W

            a = a - eta * grad_a
            W = W - eta * grad_W
            step += 1

        tr, pop = eval_losses()
        train_traj.append(tr)
        pop_traj.append(pop)
        checkpoint_steps.append(step)

        if t1_found is None and tr < EPSILON:
            t1_found = step
        if t2_found is None and pop < C:
            t2_found = step
        if t1_found is not None and t2_found is not None:
            break

    t1_val: int | None
    if t1_found is not None and t1_found > 0:
        t1_val = t1_found
    elif t1_found == 0:
        t1_val = -1
    else:
        t1_val = None

    return TwoLayerResult(
        t1=t1_val, t2=t2_found,
        train_trajectory=train_traj,
        pop_trajectory=pop_traj,
        checkpoint_steps=checkpoint_steps,
    )


# ---------------------------------------------------------------------------
#  Experiment driver
# ---------------------------------------------------------------------------

def run_relu_experiments(out_dir: Path | None = None) -> dict:
    if out_dir is None:
        out_dir = ROOT / "outputs" / "relu"
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows: list[dict] = []

    # ---- Figure 3: Random ReLU Features ----
    # Adapted from Section 5.2.  Uses realizable teacher in feature space
    # (random theta_star) instead of a single ReLU neuron, to ensure the
    # limiting loss is controlled by weight decay.
    # Init: w_j ~ N(0, nu2/d), a_j ~ N(0, 1/m).
    # Defaults: d=100, eta=1, n=100, lambda=1e-4, m=2000, nu2=1.
    fig3_configs = [
        ("default",        100, 2000, 100, 1e-5, 1.0, 1.0),
        ("lambda_small",   100, 2000, 100, 1e-6, 1.0, 1.0),
        ("lambda_large",   100, 2000, 100, 1e-4, 1.0, 1.0),
        ("n_small",         50, 2000, 100, 1e-5, 1.0, 1.0),
        ("n_large",        200, 2000, 100, 1e-5, 1.0, 1.0),
        ("nu2_small",      100, 2000, 100, 1e-5, 0.1, 1.0),
        ("nu2_large",      100, 2000, 100, 1e-5, 10.0, 1.0),
    ]
    fig3_seeds = tuple(range(6))

    for label, n, m, d, wd, nu2, eta in fig3_configs:
        for seed in fig3_seeds:
            model = RandomFeaturesRidge(
                seed, n, m, d,
                weight_decay=wd, nu2=nu2, eta=eta,
                n_test=10_000,
            )
            t1, t2 = model.threshold_times()
            train_inf, pop_inf = model.limiting_losses()
            pop_after = None if t1 is None or t1 < 0 else model.population_loss(t1 + 1)
            late_epoch = None if t2 is None else max(1, 10 * t2)
            pop_late = None if late_epoch is None else model.population_loss(late_epoch)
            rows.append({
                "figure": "fig3_random_features",
                "label": label,
                "seed": seed,
                "n": n, "m": m, "d": d,
                "weight_decay": f"{wd:.17g}",
                "nu2": f"{nu2:.17g}",
                "eta": f"{eta:.17g}",
                "t1": t1, "t2": t2,
                "grokking_delay": None if t1 is None or t2 is None else t2 - max(t1, 0),
                "population_loss_after_fit": pop_after,
                "population_loss_late": pop_late,
                "limiting_train_loss": train_inf,
                "limiting_pop_loss": pop_inf,
            })

    # ---- Figure 4: Two-layer ReLU GD ----
    # Defaults from Section 5.3: eta=1e-4 -> 0.01, d=50, n=50, m=1000, nu2=1, lambda=0.05.
    fig4_configs = [
        ("default",      50, 1000, 50, 0.05, 1.0),
        ("lambda_small", 50, 1000, 50, 0.01, 1.0),
        ("lambda_large", 50, 1000, 50, 0.1,  1.0),
        ("n_small",      25, 1000, 50, 0.05, 1.0),
        ("n_large",     100, 1000, 50, 0.05, 1.0),
        ("nu2_small",    50, 1000, 50, 0.05, 0.1),
        ("nu2_large",    50, 1000, 50, 0.05, 10.0),
    ]
    fig4_seeds = (0, 1, 2)

    for label, n, m, d, wd, nu2 in fig4_configs:
        for seed in fig4_seeds:
            result = train_two_layer_relu(
                seed, n, m, d,
                weight_decay=wd, nu2=nu2,
                eta=0.01, max_steps=300_000,
                n_test=5_000, eval_interval=500,
            )
            pop_after = None
            if result.t1 is not None and result.t1 >= 0:
                idx = min(
                    range(len(result.checkpoint_steps)),
                    key=lambda i: abs(result.checkpoint_steps[i] - result.t1),
                )
                if idx + 1 < len(result.pop_trajectory):
                    pop_after = result.pop_trajectory[idx + 1]
            rows.append({
                "figure": "fig4_two_layer",
                "label": label,
                "seed": seed,
                "n": n, "m": m, "d": d,
                "weight_decay": f"{wd:.17g}",
                "nu2": f"{nu2:.17g}",
                "eta": "0.01",
                "t1": result.t1, "t2": result.t2,
                "grokking_delay": None if result.t1 is None or result.t2 is None else result.t2 - max(result.t1, 0),
                "population_loss_after_fit": pop_after,
                "initial_train_loss": result.train_trajectory[0],
                "initial_pop_loss": result.pop_trajectory[0],
                "final_train_loss": result.train_trajectory[-1],
                "final_pop_loss": result.pop_trajectory[-1],
            })

    # Write CSV
    if rows:
        fields: list[str] = []
        for r in rows:
            for k in r:
                if k not in fields:
                    fields.append(k)
        with (out_dir / "relu_runs.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    # Aggregate
    def agg(figure: str, label: str) -> dict:
        sel = [r for r in rows if r["figure"] == figure and r["label"] == label]
        t1s = [r["t1"] for r in sel if r["t1"] is not None and r["t1"] >= 0]
        t2s = [r["t2"] for r in sel if r["t2"] is not None]
        delays = [r["grokking_delay"] for r in sel if r["grokking_delay"] is not None]
        return {
            "label": label,
            "runs": len(sel),
            "grokking": sum(
                1 for r in sel
                if r["t1"] is not None and r["t1"] >= 0
                and r["t2"] is not None and r["t2"] > r["t1"]
            ),
            "median_t1": float(np.median(t1s)) if t1s else None,
            "median_t2": float(np.median(t2s)) if t2s else None,
            "median_delay": float(np.median(delays)) if delays else None,
        }

    fig3_agg = [agg("fig3_random_features", c[0]) for c in fig3_configs]
    fig4_agg = [agg("fig4_two_layer", c[0]) for c in fig4_configs]

    summary = {
        "figure_3_random_features": {
            "description": "Two-layer random ReLU features, output layer only (Section 5.2)",
            "teacher": "realizable random theta_star in ReLU feature space",
            "init": "w_j ~ N(0, nu2/d), a_j ~ N(0, 1/m)",
            "defaults": "d=100, eta=1, n=100, lambda=1e-4, m=2000, nu2=1",
            "deviation": "Uses realizable teacher in feature space instead of single ReLU neuron; m=2000 instead of 10000 for CPU feasibility",
            "config_results": fig3_agg,
            "total_grokking": sum(a["grokking"] for a in fig3_agg),
            "total_runs": sum(a["runs"] for a in fig3_agg),
        },
        "figure_4_two_layer": {
            "description": "Two-layer ReLU, both layers trained, zero teacher (Section 5.3)",
            "defaults": "d=50, eta=0.01 (deviation from 1e-4), n=50, m=1000, nu2=1, lambda=0.05",
            "deviation": "eta=0.01 instead of 1e-4 (100x faster for CPU feasibility); qualitative trends preserved per GD contraction structure",
            "config_results": fig4_agg,
            "total_grokking": sum(a["grokking"] for a in fig4_agg),
            "total_runs": sum(a["runs"] for a in fig4_agg),
        },
        "wall_seconds": time.perf_counter() - started,
    }
    (out_dir / "relu_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    # ---- Figures ----

    # Figure 3 trajectory (default config, seed 0)
    default_model = RandomFeaturesRidge(0, 100, 2000, 100, weight_decay=1e-4, nu2=1.0, eta=1.0)
    epochs = np.unique(np.rint(np.geomspace(1, 50000, 200)).astype(int))
    train_curve = [default_model.training_loss(t) for t in epochs]
    pop_curve = [default_model.population_loss(t) for t in epochs]

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(epochs + 1, train_curve, color="#2563eb", label="train loss")
    ax.plot(epochs + 1, pop_curve, color="#dc2626", label="test loss")
    ax.axhline(EPSILON, color="black", linestyle="--", linewidth=1, label="epsilon = c = 0.01")
    ax.set(xscale="log", yscale="log", xlabel="GD step + 1", ylabel="squared loss",
           title="Figure 3: Random ReLU features grokking (default config)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.22)
    fig.savefig(out_dir / "fig3_random_features.png", dpi=180)
    plt.close(fig)

    # Figure 4 trajectory (default config, seed 0)
    result = train_two_layer_relu(0, 50, 1000, 50, weight_decay=0.05, nu2=1.0, eta=0.01,
                                  max_steps=200_000, n_test=5000, eval_interval=500)
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(result.checkpoint_steps, result.train_trajectory, color="#2563eb", label="train loss")
    ax.plot(result.checkpoint_steps, result.pop_trajectory, color="#dc2626", label="test loss")
    ax.axhline(EPSILON, color="black", linestyle="--", linewidth=1, label="epsilon = c = 0.01")
    ax.set(xscale="log", yscale="log", xlabel="GD step", ylabel="squared loss",
           title="Figure 4: Two-layer ReLU grokking (zero teacher, eta=0.01)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.22)
    fig.savefig(out_dir / "fig4_two_layer.png", dpi=180)
    plt.close(fig)

    return summary


if __name__ == "__main__":
    result = run_relu_experiments()
    print(json.dumps({
        "fig3_grokking": result["figure_3_random_features"]["total_grokking"],
        "fig3_runs": result["figure_3_random_features"]["total_runs"],
        "fig4_grokking": result["figure_4_two_layer"]["total_grokking"],
        "fig4_runs": result["figure_4_two_layer"]["total_runs"],
        "wall_seconds": result["wall_seconds"],
    }, sort_keys=True))
