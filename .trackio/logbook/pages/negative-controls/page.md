# Negative controls


---
<!-- trackio-cell
{"type": "code", "id": "cell_3d4c7b6597ab", "created_at": "2026-07-16T15:21:57+00:00", "title": "Run: python (exit 0)", "command": ["python", "-m", "unittest", "-v", "repro.tests.test_reproduction"], "exit_code": 0, "duration_s": 0.37}
-->
````bash
$ python -m unittest -v repro.tests.test_reproduction
````

exit 0 · 0.4s


````output
test_equation8_bounds_hold_for_default_draw (repro.tests.test_reproduction.RidgeDynamicsTests.test_equation8_bounds_hold_for_default_draw) ... ok
test_exact_population_formula_matches_large_monte_carlo (repro.tests.test_reproduction.RidgeDynamicsTests.test_exact_population_formula_matches_large_monte_carlo) ... ok
test_problem_generation_is_bitwise_deterministic (repro.tests.test_reproduction.RidgeDynamicsTests.test_problem_generation_is_bitwise_deterministic) ... ok
test_spectral_state_matches_explicit_gradient_descent (repro.tests.test_reproduction.RidgeDynamicsTests.test_spectral_state_matches_explicit_gradient_descent) ... ok
test_threshold_semantics_are_integer_exact (repro.tests.test_reproduction.RidgeDynamicsTests.test_threshold_semantics_are_integer_exact) ... ok

----------------------------------------------------------------------
Ran 5 tests in 0.250s

OK

````
