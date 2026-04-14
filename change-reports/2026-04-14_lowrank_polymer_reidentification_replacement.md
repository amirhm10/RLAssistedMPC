# 2026-04-14 low-rank polymer reidentification replacement

This change replaces the internal polymer reidentification method while preserving the public surface name `reidentification`.

## What changed

- replaced the old block-polymer helper logic in `utils/reidentification.py` with:
  - offline low-rank basis extraction from canonical polymer baseline data
  - basis caching under `Polymer/Data/`
  - online low-rank coefficient identification
  - dual-eta action mapping and smoothing
  - optional episodic observer refresh helpers
- replaced `utils/reidentification_runner.py` with the low-rank rollout:
  - separate `eta_A` and `eta_B`
  - low-rank prediction-model reconstruction
  - observer refresh logging
  - low-rank basis metadata and singular-value logging
- updated `systems/polymer/notebook_params.py` so the `reidentification` family now defaults to:
  - `basis_family = "lowrank_polymer"`
  - `rank_A = 6`, `rank_B = 2`
  - offline basis windowing and ridge defaults
  - dual eta smoothing constants
  - `observer_refresh_enabled = False`
- updated `RL_assisted_MPC_reidentification_unified.ipynb` to use:
  - `ACTION_DIM = 2`
  - low-rank config summary fields
  - observer refresh controls
  - repo/data/baseline path handoff to the runner
- updated `plot_reidentification_results(...)` so the standard surface now emits:
  - outputs
  - inputs
  - rewards
  - dual-eta traces
  - active model-delta plots
  - observer-refresh timeline
  - debug-only coefficient, delta, and singular-value plots
- replaced `report/06_polymer_reidentification_final_method.tex` with `report/07_polymer_lowrank_reidentification_final.tex`

## Notes

- The public names did not change:
  - `get_polymer_notebook_defaults("reidentification")`
  - `run_reidentification_supervisor(...)`
  - `plot_reidentification_results(...)`
- Nominal mode is still supported in code, but polymer defaults remain `disturb`.
- Observer refresh exists but defaults to off.
