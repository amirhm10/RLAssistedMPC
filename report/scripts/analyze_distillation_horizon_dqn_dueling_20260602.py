"""Regenerate distillation DQN and dueling-horizon diagnostics for 2026-06-02.

This wrapper reuses the June 1 analysis logic but writes into a new dated
artifact directory so earlier figures and summaries are preserved.
"""

from __future__ import annotations

from pathlib import Path

import analyze_distillation_horizon_agents_20260601 as horizon_analysis


ROOT = Path(__file__).resolve().parents[2]
horizon_analysis.OUT_DIR = (
    ROOT / "report" / "figures" / "distillation_horizon_dqn_dueling_20260602"
)


if __name__ == "__main__":
    horizon_analysis.main()
