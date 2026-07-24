"""Deterministic numerical tests for the independent reproduction."""

from __future__ import annotations

import unittest

import numpy as np

from repro.src.ridge_dynamics import GaussianRidgeBasis


class RidgeDynamicsTests(unittest.TestCase):
    def test_spectral_state_matches_explicit_gradient_descent(self) -> None:
        basis = GaussianRidgeBasis(seed=81, n=12, m=40)
        model = basis.model(weight_decay=0.002, nu2=1.7, eta=1.0)
        theta = model.theta0.copy()
        checkpoints = {0, 1, 2, 7, 25}
        for t in range(26):
            if t in checkpoints:
                np.testing.assert_allclose(theta, model.state(t), rtol=2e-12, atol=2e-12)
                residual = basis.x @ (theta - basis.theta_star)
                explicit_train = float(residual @ residual / (2 * basis.n))
                self.assertAlmostEqual(explicit_train, model.training_loss(t), places=12)
            gradient = basis.x.T @ (basis.x @ (theta - basis.theta_star)) / basis.n
            gradient += model.weight_decay * theta
            theta = theta - model.eta * gradient

    def test_exact_population_formula_matches_large_monte_carlo(self) -> None:
        basis = GaussianRidgeBasis(seed=13, n=20, m=80)
        model = basis.model(weight_decay=0.001, nu2=1.0, eta=1.0)
        state = model.state(300)
        rng = np.random.default_rng(901)
        x_test = rng.normal(0.0, 1.0 / np.sqrt(basis.m), size=(120_000, basis.m))
        residual = x_test @ (state - basis.theta_star)
        monte_carlo = float(np.mean(residual * residual))
        exact = model.population_loss(300)
        self.assertLess(abs(monte_carlo - exact), 0.015 * exact)

    def test_threshold_semantics_are_integer_exact(self) -> None:
        basis = GaussianRidgeBasis(seed=5, n=40, m=300)
        model = basis.model(weight_decay=1e-4, nu2=1.0, eta=1.0)
        times = model.threshold_times(0.01, 0.01)
        self.assertIsNotNone(times.t1)
        self.assertIsNotNone(times.t2)
        assert times.t1 is not None and times.t2 is not None
        self.assertGreaterEqual(model.training_loss(times.t1), 0.01)
        self.assertLess(model.training_loss(times.t1 + 1), 0.01)
        self.assertGreater(model.population_loss(times.t2 - 1), 0.01)
        self.assertLessEqual(model.population_loss(times.t2), 0.01 + 1e-14)
        self.assertTrue(times.training_monotone_to_crossing)
        self.assertTrue(times.population_monotone_to_crossing)

    def test_equation8_bounds_hold_for_default_draw(self) -> None:
        basis = GaussianRidgeBasis(seed=2, n=100, m=1000)
        model = basis.model(weight_decay=1e-4, nu2=1.0, eta=1.0)
        times = model.threshold_times()
        t1_upper, t2_lower = model.equation8_bounds()
        assert times.t1 is not None and times.t2 is not None
        self.assertLessEqual(times.t1, t1_upper)
        self.assertGreaterEqual(times.t2, t2_lower)

    def test_problem_generation_is_bitwise_deterministic(self) -> None:
        first = GaussianRidgeBasis(seed=91, n=16, m=64)
        second = GaussianRidgeBasis(seed=91, n=16, m=64)
        self.assertTrue(np.array_equal(first.x, second.x))
        self.assertTrue(np.array_equal(first.theta_star, second.theta_star))
        self.assertTrue(np.array_equal(first.standard_init, second.standard_init))
        model_a = first.model(weight_decay=1e-4, nu2=10.0, eta=1.0)
        model_b = second.model(weight_decay=1e-4, nu2=10.0, eta=1.0)
        self.assertEqual(model_a.threshold_times(), model_b.threshold_times())

    def test_arbitrary_teacher_types_produce_grokking(self) -> None:
        for kind in ["random", "sparse", "one_hot", "uniform", "top_k"]:
            for norm in [0.5, 1.0, 2.0]:
                basis = GaussianRidgeBasis(
                    seed=7, n=100, m=1000,
                    teacher_kind=kind, teacher_norm=norm,
                )
                self.assertAlmostEqual(
                    float(np.linalg.norm(basis.theta_star)), norm, places=12,
                )
                model = basis.model(weight_decay=1e-4, nu2=1.0, eta=1.0)
                times = model.threshold_times(0.01, 0.01)
                self.assertIsNotNone(times.t1)
                self.assertIsNotNone(times.t2)
                assert times.t1 is not None and times.t2 is not None
                self.assertGreater(times.t2, times.t1)
                self.assertLess(model.training_loss(times.t1 + 1), 0.01)
                self.assertGreater(model.population_loss(times.t1 + 1), 0.01)
                self.assertLess(model.population_loss(10 * times.t2), 0.01)

    def test_random_features_relu_grokking(self) -> None:
        from repro.src.relu_experiments import RandomFeaturesRidge
        model = RandomFeaturesRidge(
            seed=0, n=100, m=2000, d=100,
            weight_decay=1e-4, nu2=1.0, eta=1.0, n_test=2000,
        )
        t1, t2 = model.threshold_times(max_steps=500000)
        self.assertIsNotNone(t1)
        self.assertIsNotNone(t2)
        assert t1 is not None and t2 is not None
        self.assertGreaterEqual(t1, 0)
        self.assertGreater(t2, t1)
        self.assertLess(model.training_loss(t1 + 1), 0.01)
        self.assertGreater(model.population_loss(t1 + 1), 0.01)


if __name__ == "__main__":
    unittest.main(verbosity=2)

