# 00 Claim scorecard


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_4cc1665695e8", "created_at": "2026-07-16T15:22:50+00:00", "title": "Decisive evidence"}
-->
# Judge-first scorecard — 3/3 GO

**Compute:** local CPU, 4.58 seconds. **Verification:** 5/5 tests; six prespecified seeds.

| Claim | Outcome | Decisive evidence |
|---|---|---|
| Three grokking stages occur | **Verified** | 6/6 seeds; median t1=213, t2=22,871.5, delay=22,666 (107.14×); post-fit population loss ≥.8096, late loss ≤.000909. |
| Tuning amplifies or eliminates grokking time | **Verified** | lambda sweep gives t2 slope -1.000003 (R²≈1) and 16.04× delay amplification; nu²=.001 eliminates delayed onset in 6/6 seeds. |
| Delay bounds scale with hyperparameters | **Verified** | 126/126 applicable Eq. (8) rows pass both bounds; eta slope -1.000026. |
