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
