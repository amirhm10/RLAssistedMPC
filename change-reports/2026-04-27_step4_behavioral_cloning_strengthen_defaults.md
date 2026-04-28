# 2026-04-27 Step 4 Behavioral Cloning Strengthen Defaults

Updated the polymer matrix and structured-matrix Step 4 behavioral-cloning defaults and rewrote the Step 4 framing in the matrix multiplier recovery report.

Code changes:

- `systems/polymer/notebook_params.py`
  - made behavioral-cloning defaults parameterized
  - increased polymer scalar BC from `lambda_bc_start = 0.1`, `active_subepisodes = 10` to `0.3`, `20`
  - increased polymer structured BC from `0.1`, `10` to `0.6`, `25`

Report changes:

- `report/matrix_multiplier_cap_calculation_and_distillation_recovery.md`
  - renamed the old “Implementation Progress: Shared BC Next Step” section to Step 4
  - added Step 4A through Step 4G wording for the BC follow-on options
  - documented that Step 4B, 4C, and 4D are now implemented in polymer defaults

Interpretation:

- scalar polymer gets a stronger and longer nominal anchor,
- structured polymer gets an even stronger anchor because the BC-only run showed a larger off-nominal handoff problem,
- distillation BC remains disabled by default until the stronger polymer Step 4 rerun is reviewed.
