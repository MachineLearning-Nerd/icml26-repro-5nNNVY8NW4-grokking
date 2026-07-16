#!/usr/bin/env python3
"""Fail-closed verification of metrics, source identity, CPU provenance, and hashes."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "full"


def sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def main() -> None:
    checks = 0

    claims = json.loads((OUT / "claims_snapshot.json").read_text())
    assert claims["openreview_id"] == "5nNNVY8NW4"
    assert claims["tags"] == ["icml2026-repro", "paper-5nNNVY8NW4"]
    assert len(claims["claims"]) == 3
    checks += 1

    audit = json.loads((OUT / "source_audit.json").read_text())
    primary = audit["primary_source"]
    assert primary["pdf_sha256"] == sha256(ROOT / "paper_2601.19791v3.pdf")
    assert primary["source_tar_sha256"] == sha256(ROOT / "source/arxiv/2601.19791v3.tar")
    assert primary["main_tex_sha256"] == sha256(ROOT / "source/arxiv/grokking_ridge_regression.tex")
    checks += 1

    code_audit = audit["official_code_audit"]
    assert code_audit["paper_links_code_or_repository"] is False
    assert code_audit["exact_title_github_repository_results"] == 0
    assert code_audit["arxiv_id_github_repository_results"] == 0
    checks += 1

    summary = json.loads((OUT / "summary.json").read_text())
    with (OUT / "runs.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == summary["protocol"]["total_config_seed_runs"] == 132
    assert sorted({int(row["seed"]) for row in rows}) == list(range(6))
    assert {int(row["m"]) for row in rows if row["sweep"] == "dimension"} == {1000, 2000, 3000}
    checks += 1

    numeric_fields = [
        "initial_training_loss", "initial_population_loss", "limiting_training_loss",
        "limiting_population_loss", "equation8_t1_upper", "spectral_orthogonality_error",
    ]
    assert all(math.isfinite(float(row[field])) for row in rows for field in numeric_fields)
    assert max(float(row["spectral_orthogonality_error"]) for row in rows) < 2e-14
    assert all(row["training_monotone_to_crossing"] == "True" for row in rows)
    assert all(row["population_monotone_to_crossing"] == "True" for row in rows)
    checks += 1

    c1 = summary["claim_1"]
    assert c1["seeds_passing_all_three_stages"] == c1["seeds"] == 6
    assert c1["median_delay"] > 22_000
    assert c1["median_t2_over_t1_plus_one"] > 100
    assert c1["minimum_population_loss_after_fit"] > 0.80
    assert c1["maximum_late_population_loss"] < 0.001
    assert c1["maximum_limiting_population_loss"] < 0.001
    checks += 1

    dim_fit = c1["dimension_vs_limiting_loss_loglog"]
    assert -1.05 < dim_fit["slope"] < -0.85
    assert dim_fit["r_squared"] > 0.999
    checks += 1

    c2 = summary["claim_2"]
    weight_fit = c2["weight_decay_t2_loglog"]
    assert -1.01 < weight_fit["slope"] < -0.99
    assert weight_fit["r_squared"] > 0.99999
    assert c2["delay_amplification_smallest_vs_largest_lambda"] > 15.5
    assert c2["elimination_seeds_with_t1_empty_and_t2_zero"] == c2["elimination_seeds"] == 6
    assert c2["initialization_t2_vs_log_nu2"]["r_squared"] > 0.99999
    checks += 1

    c3 = summary["claim_3"]
    assert c3["qualifying_config_seed_rows"] == 126
    assert c3["t1_upper_bound_passes"] == 126
    assert c3["t2_lower_bound_passes"] == 126
    assert c3["minimum_observed_to_t2_lower_ratio"] >= 1.0
    assert c3["maximum_observed_to_t1_upper_ratio"] <= 1.0
    assert -1.01 < c3["learning_rate_t2_loglog"]["slope"] < -0.99
    checks += 1

    env = json.loads((OUT / "environment.json").read_text())
    assert env["gpu_used"] is False and env["paid_service_used"] is False
    assert env["cuda_visible_devices"] == ""
    assert all(value == "1" for value in env["thread_limits"].values())
    checks += 1

    manifest_lines = (OUT / "CHECKSUMS.sha256").read_text().splitlines()
    verified_hashes = 0
    for line in manifest_lines:
        expected, rel = line.split("  ", 1)
        path = ROOT / rel
        assert path.is_file(), rel
        assert sha256(path) == expected, rel
        verified_hashes += 1
    assert verified_hashes >= 35
    checks += 1

    required = [
        OUT / "runs.csv", OUT / "aggregates.csv", OUT / "trajectories.csv",
        OUT / "three_stages.png", OUT / "hyperparameter_scaling.png",
        OUT / "summary.json", OUT / "source_audit.json", OUT / "environment.json",
    ]
    assert all(path.stat().st_size > 0 for path in required)
    checks += 1

    print(
        f"PASS: {checks} independent gates + {verified_hashes} SHA-256 files; "
        "132 CPU-only runs; three scored claims verified within stated scope"
    )


if __name__ == "__main__":
    main()


