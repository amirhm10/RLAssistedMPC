# Option 1 Offline Multiplier Sensitivity

Implemented the diagnostic-only rho and finite-horizon gain scan for scalar and structured matrix multiplier experiments.

- Added `utils/multiplier_sensitivity.py` with spectral-radius, Markov-gain, local log-sensitivity, random candidate scan, advisory-bound, CSV, and optional plot helpers.
- Added offline diagnostic defaults to polymer and distillation matrix/structured matrix configs.
- Wired all four unified matrix notebooks with a pre-training diagnostic cell.
- Kept polymer diagnostics enabled by default and distillation diagnostics disabled by default.
- Updated the ongoing matrix-cap report with an Option 1 progress scheme.

Validation:

- Imported the new diagnostic module under `rl-env`.
- Ran synthetic scalar and structured diagnostics.
- Validated the four edited notebook JSON files.
- Checked polymer/distillation diagnostic defaults.
- Ran a small polymer repo-data smoke diagnostic for matrix and structured matrix without training.
