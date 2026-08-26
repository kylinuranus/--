from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
STRATEGIES_DIR = PROJECT_ROOT / "strategies"
SCREENING_RESULTS_DIR = PROJECT_ROOT / "screening_results"
WATCHLISTS_DIR = PROJECT_ROOT / "watchlists"
LOGS_DIR = PROJECT_ROOT / "logs"


def ensure_project_dirs() -> None:
    for path in [STRATEGIES_DIR, SCREENING_RESULTS_DIR, WATCHLISTS_DIR, LOGS_DIR]:
        path.mkdir(parents=True, exist_ok=True)
