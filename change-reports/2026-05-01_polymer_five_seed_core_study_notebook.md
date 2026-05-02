## Summary

Added a new polymer five-seed core study notebook and a shared Python helper module to run the live polymer RL methods across fixed seeds, save native per-run artifacts, import the canonical baseline MPC save, and export aggregate slide-ready analysis outputs.

## Added

- `polymer_five_seed_core_study.ipynb`
- `utils/polymer_multiseed_core_study.py`

## Scope

- Methods included: `horizon_dueling`, `matrix`, `weights`, `residual`, `combined`
- Continuous agents restricted to TD3
- Discrete agent restricted to dueling DQN
- Combined study overrides the horizon subagent to dueling DQN while keeping matrix/weights/residual on TD3
- Baseline MPC is loaded from the canonical polymer `Data/` pickle and is never rerun by the study notebook

## Outputs

The helper exports:

- per-run native result bundles and comparison plots through the existing plotting layer
- a flat CSV summary
- a slide-metrics CSV
- a JSON manifest
- Markdown summaries
- aggregate figures under `report/figures/...`

## Validation

- Imported the new helper and validated the default study config
- Loaded the polymer common system setup through the new helper and confirmed the model and scenario shapes
- Validated notebook JSON structure by loading the generated `.ipynb` file

## Notes

The full five-seed study was not executed during implementation because it would launch 25 full training/evaluation runs.
