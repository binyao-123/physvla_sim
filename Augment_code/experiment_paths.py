"""Default paths for FITR VLM / bench experiment artifacts."""

from __future__ import annotations

from pathlib import Path

AUGMENT_ROOT = Path(__file__).resolve().parent
EXPERIMENTS_DIR = AUGMENT_ROOT / "experiments"
EVAL_DIR = EXPERIMENTS_DIR / "eval"

DEFAULT_VLM_RESULTS = EXPERIMENTS_DIR / "vlm_results.jsonl"
DEFAULT_VLM_RESULTS_BASELINE = EXPERIMENTS_DIR / "vlm_results_baseline.jsonl"
DEFAULT_VLM_AFFORDANCE_RESULTS = EXPERIMENTS_DIR / "vlm_affordance_results.jsonl"


def ensure_experiment_dirs() -> None:
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)


def eval_report_path(name: str) -> Path:
    """Build path under experiments/eval/ (add .json if missing)."""
    ensure_experiment_dirs()
    stem = name[:-5] if name.endswith(".json") else name
    return EVAL_DIR / f"{stem}.json"
