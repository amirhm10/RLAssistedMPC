## Summary

Added the canonical Van de Vusse `baseline` workflow for offset-free MPC.

## What Changed

- Added the notebook-facing `baseline` family to `systems/vandevusse/notebook_params.py`.
- Added Van de Vusse baseline constants, run profiles, setpoint schedules, and disturbance-profile definitions in `systems/vandevusse/config.py`.
- Added baseline disturbance-schedule helpers in `systems/vandevusse/scenarios.py`.
- Added canonical baseline artifact path helpers in `systems/vandevusse/data_io.py`.
- Added the reusable baseline runtime in `systems/vandevusse/baseline_mpc.py`.
- Exported the new baseline helpers from `systems/vandevusse/__init__.py`.
- Added the thin root notebook `vandevusse_MPCOffsetFree_unified.ipynb`.
- Added the baseline report `report/10_vandevusse_offset_free_mpc_baseline.tex`.

## Canonical Baseline

- Notebook family: `get_vandevusse_notebook_defaults("baseline")`
- Notebook entrypoint: `vandevusse_MPCOffsetFree_unified.ipynb`
- Canonical disturbed profile: `ca0_blocks`
- Canonical saved pickles:
  - `mpc_results_nominal.pickle`
  - `mpc_results_disturb_ca0_blocks.pickle`

## Disturbance Policy

The disturbed baseline varies only the feed concentration:

- `c_A0 = [5.10, 4.70, 5.50, 5.10]`
- `T_in = [378.1, 378.1, 378.1, 378.1]`

## Scope

- This adds the Van de Vusse offset-free MPC baseline only.
- RL layers for Van de Vusse remain out of scope in this change.
- Polymer and distillation public surfaces were not changed.
