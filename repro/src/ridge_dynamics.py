"""Exact CPU spectral dynamics for the paper's Gaussian ridge experiment.

The implementation follows Eq. (1)--(2) of arXiv:2601.19791v3.  It does not
approximate gradient descent: an eigendecomposition of X X^T lets us evaluate
the integer-time full-batch GD iterate in closed form, including its component
orthogonal to the training-data span.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class ThresholdResult:
    t1: int | None
    t2: int | None
    training_monotone_to_crossing: bool
    population_monotone_to_crossing: bool


class GaussianRidgeBasis:
    """A deterministic problem draw that can be reused across hyperparameters."""

    def __init__(self, seed: int, n: int, m: int) -> None:
        if not 0 < n < m:
            raise ValueError("the reproduction requires an overparameterized n < m problem")
        self.seed = int(seed)
        self.n = int(n)
        self.m = int(m)

        data_ss, teacher_ss, init_ss = np.random.SeedSequence(seed).spawn(3)
        data_rng = np.random.default_rng(data_ss)
        teacher_rng = np.random.default_rng(teacher_ss)
        init_rng = np.random.default_rng(init_ss)

        self.x = data_rng.normal(0.0, 1.0 / math.sqrt(m), size=(n, m))
        teacher = teacher_rng.normal(size=m)
        self.theta_star = teacher / np.linalg.norm(teacher)
        self.standard_init = init_rng.normal(size=m)

        gram = self.x @ self.x.T
        eigvals, u = np.linalg.eigh(gram)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        u = u[:, order]
        tol = np.finfo(float).eps * max(n, m) * eigvals[0]
        if np.count_nonzero(eigvals > tol) != n:
            raise RuntimeError("Gaussian design unexpectedly lacks full row rank")
        singular = np.sqrt(eigvals)
        self.vt = (u.T @ self.x) / singular[:, None]
        self.eig_a = eigvals / n
        self.singular_sq = eigvals
        self.orthogonality_error = float(
            np.max(np.abs(self.vt @ self.vt.T - np.eye(n)))
        )

        self.teacher_row = self.vt @ self.theta_star
        self.teacher_null = self.theta_star - self.vt.T @ self.teacher_row
        self.init_row_standard = self.vt @ self.standard_init
        self.init_null_standard = self.standard_init - self.vt.T @ self.init_row_standard

    def model(self, *, weight_decay: float, nu2: float, eta: float) -> "SpectralRidgeGD":
        return SpectralRidgeGD(self, weight_decay=weight_decay, nu2=nu2, eta=eta)


class SpectralRidgeGD:
    """Closed-form integer-time full-batch GD for one ridge configuration."""

    def __init__(
        self,
        basis: GaussianRidgeBasis,
        *,
        weight_decay: float,
        nu2: float,
        eta: float,
    ) -> None:
        if weight_decay <= 0 or nu2 <= 0 or eta <= 0:
            raise ValueError("weight_decay, nu2, and eta must be positive")
        self.basis = basis
        self.weight_decay = float(weight_decay)
        self.nu2 = float(nu2)
        self.eta = float(eta)

        scale = math.sqrt(nu2)
        self.theta0 = scale * basis.standard_init
        self.alpha0 = scale * basis.init_row_standard
        self.theta0_null = scale * basis.init_null_standard
        self.beta = basis.teacher_row
        self.teacher_null = basis.teacher_null

        self.q = 1.0 - eta * (basis.eig_a + weight_decay)
        self.r = 1.0 - eta * weight_decay
        if not (0.0 < self.r < 1.0) or np.any(self.q <= 0.0) or np.any(self.q >= 1.0):
            raise ValueError("configuration is outside the positive-contraction GD regime")
        self.alpha_inf = (basis.eig_a / (basis.eig_a + weight_decay)) * self.beta

        self.null_init_sq = float(self.theta0_null @ self.theta0_null)
        self.null_teacher_sq = float(self.teacher_null @ self.teacher_null)
        self.null_cross = float(self.theta0_null @ self.teacher_null)

    @property
    def n(self) -> int:
        return self.basis.n

    @property
    def m(self) -> int:
        return self.basis.m

    def row_coordinates(self, t: int) -> np.ndarray:
        if t < 0:
            raise ValueError("t must be nonnegative")
        qt = np.power(self.q, t)
        return qt * self.alpha0 + (1.0 - qt) * self.alpha_inf

    def state(self, t: int) -> np.ndarray:
        alpha = self.row_coordinates(t)
        return self.basis.vt.T @ alpha + (self.r**t) * self.theta0_null

    def training_loss(self, t: int) -> float:
        diff = self.row_coordinates(t) - self.beta
        return float(np.dot(self.basis.singular_sq, diff * diff) / (2.0 * self.n))

    def population_loss(self, t: int) -> float:
        """Exact population loss for x ~ N(0, I/m), without Monte Carlo."""
        row_diff = self.row_coordinates(t) - self.beta
        rt = self.r**t
        null_sq = (
            rt * rt * self.null_init_sq
            - 2.0 * rt * self.null_cross
            + self.null_teacher_sq
        )
        return float((np.dot(row_diff, row_diff) + null_sq) / self.m)

    def regularized_objective(self, t: int) -> float:
        alpha = self.row_coordinates(t)
        rt = self.r**t
        theta_sq = float(np.dot(alpha, alpha) + rt * rt * self.null_init_sq)
        return self.training_loss(t) + 0.5 * self.weight_decay * theta_sq

    def limiting_losses(self) -> tuple[float, float]:
        row_diff = self.alpha_inf - self.beta
        train = float(np.dot(self.basis.singular_sq, row_diff * row_diff) / (2.0 * self.n))
        population = float((np.dot(row_diff, row_diff) + self.null_teacher_sq) / self.m)
        return train, population

    @staticmethod
    def _monotone_certificate(fn: Callable[[int], float], hi: int) -> bool:
        if hi <= 1:
            return True
        grid = np.unique(np.rint(np.linspace(0, hi, 1025)).astype(np.int64))
        values = np.asarray([fn(int(t)) for t in grid])
        tolerance = 1e-11 * max(1.0, float(np.max(np.abs(values))))
        return bool(np.all(np.diff(values) <= tolerance))

    @staticmethod
    def _first_below(
        fn: Callable[[int], float], threshold: float, max_steps: int
    ) -> tuple[int | None, bool]:
        if fn(0) < threshold:
            return 0, True
        hi = 1
        while hi <= max_steps and fn(hi) >= threshold:
            hi *= 2
        if hi > max_steps:
            return None, False
        monotone = SpectralRidgeGD._monotone_certificate(fn, hi)
        lo = hi // 2
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if fn(mid) < threshold:
                hi = mid
            else:
                lo = mid
        return hi, monotone

    def threshold_times(
        self, epsilon: float = 0.01, c: float = 0.01, max_steps: int = 100_000_000
    ) -> ThresholdResult:
        train_cross, train_monotone = self._first_below(
            self.training_loss, epsilon, max_steps
        )
        pop_cross, pop_monotone = self._first_below(
            lambda t: self.population_loss(t) + np.finfo(float).eps,
            c + np.finfo(float).eps,
            max_steps,
        )
        # If already below epsilon at initialization, the defining set for t1 is empty.
        t1 = -1 if train_cross == 0 else (None if train_cross is None else train_cross - 1)
        return ThresholdResult(t1, pop_cross, train_monotone, pop_monotone)

    def equation8_bounds(self, epsilon: float = 0.01) -> tuple[float, float]:
        """Paper Eq. (8): t1 upper bound and t2 lower bound."""
        t1_arg = 14.0 * self.m * self.nu2 / epsilon
        t1_upper = self.n * math.log(t1_arg) / (
            2.0 * self.eta * float(np.min(self.basis.singular_sq))
        )
        t2_arg = (self.m - self.n) * self.nu2 / (8.0 * self.m * epsilon)
        t2_lower = (
            math.log(t2_arg) / (2.02 * self.eta * self.weight_decay)
            if t2_arg > 1.0
            else math.nan
        )
        return t1_upper, t2_lower


