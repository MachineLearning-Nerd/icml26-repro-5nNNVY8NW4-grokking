# Artifact bundle


---
<!-- trackio-cell
{"type": "dashboard", "id": "cell_329977ae0201", "created_at": "2026-07-16T15:22:23+00:00", "title": "Dashboard: grokking-ridge-repro", "dashboard_project": "grokking-ridge-repro"}
-->
**🎯 Trackio dashboard** `grokking-ridge-repro`

trackio-local-dashboard://grokking-ridge-repro


---
<!-- trackio-cell
{"type": "code", "id": "cell_bd2e48211ccc", "created_at": "2026-07-16T15:22:24+00:00", "title": "Run: python log_bundle.py (exit 0)", "command": ["python", "repro/src/log_bundle.py"], "exit_code": 0, "duration_s": 0.917}
-->
````bash
$ python repro/src/log_bundle.py
````

exit 0 · 0.9s


````python title=log_bundle.py
"""Register the compact grokking evidence bundle."""

from pathlib import Path
import trackio

ROOT = Path(__file__).resolve().parents[2]

def main() -> None:
    trackio.init(project="grokking-ridge-repro", name="cpu-full", auto_log_cpu=False)
    artifact = trackio.Artifact(
        name="repro-bundle", type="dataset",
        metadata={"openreview_id": "5nNNVY8NW4", "arxiv_id": "2601.19791"},
    )
    artifact.add_dir(ROOT / "outputs" / "full", name="outputs/full")
    artifact.add_dir(ROOT / "repro", name="repro")
    trackio.log_artifact(artifact, aliases=["full"])
    trackio.finish()

if __name__ == "__main__":
    main()

````


````output
* Trackio project initialized: grokking-ridge-repro
* Trackio metrics logged to the local Trackio cache.
* View dashboard by running in your terminal:
[1m[38;5;208mtrackio show --project "grokking-ridge-repro"[0m
* or by running in Python: trackio.show(project="grokking-ridge-repro")
* Created new run: cpu-full
* Run finished. Uploading logs to Trackio (please wait...)

````
