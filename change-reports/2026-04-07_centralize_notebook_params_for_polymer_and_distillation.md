# Centralize Notebook Parameters For Polymer And Distillation

## What changed
- Added `systems/polymer/notebook_params.py` as the polymer notebook-facing source of truth.
- Added `systems/distillation/notebook_params.py` as the distillation notebook-facing source of truth.
- Updated the active polymer and distillation notebooks so their top-level run controls, path defaults, system setup values, reward defaults, controller settings, and agent defaults are loaded from those parameter modules.
- Updated `systems/polymer/__init__.py` and `systems/distillation/__init__.py` to export the new getters.

## Important behavior
- Editing the notebook parameter dictionaries now changes the notebook defaults directly on the next run.
- One-off notebook edits are still possible by changing the visible variables in the notebook config cells.
- The notebooks no longer need to carry large duplicated blocks of hard-coded defaults.

## Notes
- This change also corrected the lingering TD3/SAC constructor drift in the notebook cells that still had mixed argument wiring.
- Unrelated local data/result artifacts were left untouched.
