# Align Polymer Markov With Unified Runtime

## Summary

Aligned the polymer Markov notebook with the shared unified polymer RL runtime.

## Changes

- Added `POLYMER_MARKOV_DEFAULTS` and the `"markov"` notebook family in `systems/polymer/notebook_params.py`.
- Added `utils/markov_runner.py` as the shared Markov rollout entrypoint.
- Added `plot_markov_correction_results()` and `plot_markov_correction_results_core()` to the shared plotting layer.
- Replaced `polymer_markov_corrected_mpc_unified.ipynb` with a unified-style notebook that uses shared defaults, shared reward construction, the new shared runner, and the existing MPC-vs-RL comparison helper.
- Reduced `report/scripts/generate_polymer_markov_correction_assets.py` to an optional saved-bundle report writer instead of a runtime owner.

## Verification

- Parsed the rewritten notebook JSON successfully.
- Compiled the changed Python files with a no-write syntax check.
- Verified `get_polymer_notebook_defaults("markov")` resolves and uses the expected disturbed TD3 defaults.
- Ran a short RL-only Markov smoke validation through the new shared runner and plotter.
- Confirmed the smoke run wrote one Markov RL result directory and one MPC-vs-RL comparison directory.
- Confirmed the live reward path matches `make_reward_fn_relative_QR(...)` on the saved rollout data.
- Confirmed the default path produced no `phase*.png` debug outputs.
- Ran a debug-enabled short validation and confirmed the optional phase figures are saved only when explicitly enabled.
