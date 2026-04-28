# Warm-Start Report And TD3 Policy Delay Refresh

Date: 2026-04-23

## What Changed

- Expanded `report/distillation_warm_start_training_analysis.md` into a fuller technical note with:
  - explicit warm-start schedule analysis
  - TD3 objective mathematics
  - release-drop diagnostics
  - literature-backed next-step recommendations
  - exact external paper figure pointers
- Refreshed `report/scripts/distillation_warm_start_analysis.py` outputs and added literature-support / method-tradeoff figures under `report/figures/`.
- Changed TD3 `policy_delay` defaults from `4` to `2` wherever the repo still defined the notebook default surface:
  - `systems/polymer/notebook_params.py`
  - `systems/distillation/notebook_params.py`
  - `RL_assisted_MPC_Poles.ipynb`

## Why

- The warm-start report needed to move from a short code audit to a more defensible research note with equations, mechanism-level reasoning, and direct literature support.
- The TD3 default surface was inconsistent: most shared notebook families now inherit `policy_delay = 2`, but the standalone poles notebook still hardcoded `4`.

## Validation

- `python -m py_compile systems/polymer/notebook_params.py systems/distillation/notebook_params.py`
- `python -m py_compile report/scripts/distillation_warm_start_analysis.py`
- notebook JSON edit for `RL_assisted_MPC_Poles.ipynb` performed with `nbformat`
- report figures regenerated after the TD3 default refresh
