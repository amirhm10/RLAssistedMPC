## Summary

Adjusted the polymer re-identification v3 study defaults so the short diagnostic preset uses `warm_start = 4`.

## Change

Updated the v3 warm-start value in:

- `systems/polymer/notebook_params.py`
- `experiments/reid_batch_ablation_matrix.py`

The affected v3 defaults now use:

- `n_tests = 40`
- `set_points_len = 200`
- `warm_start = 4`

## Reward Defaults Check

The v3 study reward defaults did **not** require a code change.

`reid_batch_v3_study["reward"]` already matches the shared polymer RL reward defaults used by:

- `matrix`
- `structured_matrix`
- `reid_batch`
- `reid_batch_v2`

## Validation

- `python -m py_compile systems/polymer/notebook_params.py experiments/reid_batch_ablation_matrix.py`
- verified:
  - `get_polymer_notebook_defaults("reid_batch_v3_study")["episode_defaults"]["warm_start"] == 4`
  - `get_polymer_notebook_defaults("reid_batch_v3_study")["study"]["short_diagnostic_preset"]["warm_start"] == 4`
  - v3 reward defaults are equal to the shared polymer matrix reward defaults
