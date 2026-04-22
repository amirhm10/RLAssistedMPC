# 2026-04-22 Distillation Tight-A Wide-B Matrix Defaults

- Updated [systems/distillation/config.py](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/systems/distillation/config.py) so the active distillation matrix defaults use a tight `A` multiplier band and wide `B` multipliers:
  - plain matrix `alpha` now defaults to `[0.99, 1.01]`
  - plain matrix `delta` stays wide at `[0.75, 1.25]`
- Updated [systems/distillation/notebook_params.py](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/systems/distillation/notebook_params.py) so the distillation structured-matrix defaults use the same tight `A` / wide `B` pattern:
  - structured `A` overrides default to `[0.99, 1.01]`
  - structured `B` overrides stay wide at `[0.75, 1.25]`
  - `prediction_fallback_on_solve_failure` is set `True` in the structured defaults
- Updated [distillation_RL_assisted_MPC_structured_matrices_unified.ipynb](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/distillation_RL_assisted_MPC_structured_matrices_unified.ipynb) so the structured notebook carries the `A/B` overrides and fallback flag into `structured_cfg`, which keeps the shared runner-side cap refresh robust.
