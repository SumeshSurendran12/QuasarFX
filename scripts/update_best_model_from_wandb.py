from __future__ import annotations

import os
import shutil
from pathlib import Path

import wandb
from dotenv import load_dotenv


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    dotenv_path = project_root / ".env"
    load_dotenv(dotenv_path=dotenv_path, override=True)
    api_key = os.getenv("WANDB_API_KEY")
    if not api_key:
        raise SystemExit(f"WANDB_API_KEY not found in {dotenv_path}")
    if len(api_key) < 40:
        raise SystemExit(f"WANDB_API_KEY looks too short (len={len(api_key)}). Check {dotenv_path}.")

    # Resolve artifact name with env overrides
    artifact_name = os.getenv("WANDB_BEST_MODEL_ARTIFACT") or os.getenv("WANDB_ARTIFACT")
    if not artifact_name:
        entity = os.getenv("WANDB_ENTITY")
        project = os.getenv("WANDB_PROJECT")
        if entity and project:
            artifact_name = f"{entity}/{project}/best_model:latest"
        else:
            artifact_name = "sumeshsurendran12-ut-health-houston/forex-trading-bot/best_model:latest"
    artifact_type = os.getenv("WANDB_BEST_MODEL_TYPE", "model")
    out_path = Path(os.getenv("WANDB_BEST_MODEL_OUT", "models/best_model_live.zip"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Use public API to avoid requiring wandb.init() for simple artifact fetch
    api = wandb.Api()
    artifact = api.artifact(artifact_name, type=artifact_type)
    artifact_dir = Path(artifact.download())

    src = artifact_dir / "best_model.zip"
    if not src.exists():
        candidates = sorted(artifact_dir.glob("*.zip"))
        if not candidates:
            raise SystemExit(f"No .zip model found in artifact dir: {artifact_dir}")
        src = candidates[0]

    shutil.copy2(src, out_path)
    print(f"Updated: {out_path}")


if __name__ == "__main__":
    main()
