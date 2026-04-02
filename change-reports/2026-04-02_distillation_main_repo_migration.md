# Distillation Main-Repo Migration

## Summary

This change migrates the distillation column case into the main unified architecture as a first-class canonical system while keeping the old distillation subtree as a read-only archived reference.

## Added canonical distillation layer

- `systems/distillation/config.py`
- `systems/distillation/data_io.py`
- `systems/distillation/labels.py`
- `systems/distillation/plant.py`
- `systems/distillation/scenarios.py`
- `systems/distillation/system_id.py`

These modules now provide:

- Aspen/distillation plant access
- canonical distillation data/result paths
- disturbance-profile generation for `none`, `ramp`, and `fluctuation`
- distillation-specific plot labels and metadata
- system-identification helpers including the MATLAB-assisted state-space reconstruction path

## Added canonical data namespace

- `Data/distillation/system_dict.pickle`
- `Data/distillation/scaling_factor.pickle`
- `Data/distillation/min_max_states.pickle`
- `Data/distillation/Reflux.csv`
- `Data/distillation/Reboiler.csv`
- `Data/distillation/mpc_results_nominal.pickle`
- `Data/distillation/mpc_results_disturb_ramp.pickle`
- `Data/distillation/mpc_results_disturb_fluctuation.pickle`

The migration maps the old distillation naming into canonical names and keeps the archived source tree untouched.

## Added unified distillation notebook surface

- `distillation_systemIdentification_unified.ipynb`
- `distillation_MPCOffsetFree_unified.ipynb`
- `distillation_RL_assisted_MPC_horizons_unified.ipynb`
- `distillation_RL_assisted_MPC_matrices_unified.ipynb`
- `distillation_RL_assisted_MPC_weights_unified.ipynb`
- `distillation_RL_assisted_MPC_residual_unified.ipynb`
- `distillation_RL_assisted_MPC_combined_unified.ipynb`

All of these notebooks follow the polymer unified pattern:

- repo-root resolution
- top-level config cell
- shared runner usage
- shared plotting usage
- canonical save/result locations

Distillation-specific controls are:

- `RUN_MODE = "nominal" | "disturb"`
- `DISTURBANCE_PROFILE = "none" | "ramp" | "fluctuation"`

Agent-side controls are aligned with the polymer unified notebooks:

- `STATE_MODE = "standard" | "mismatch"`
- matrices/weights/residual: `AGENT_KIND = "td3" | "sac"`
- residual: `USE_RHO_AUTHORITY = True | False`
- combined: per-agent enable/disable plus per-agent kind/state settings

## Shared-layer changes

The shared helper and runner layer was generalized so distillation can reuse the same algorithms instead of forking a second runtime stack.

- `utils/helpers.py`
  - generalized data-dir resolution
  - generalized disturbance-profile conversion
  - added custom disturbance stepping support through `step_system_with_disturbance(...)`
- shared runners now accept:
  - `system_stepper`
  - `system_metadata`
  - `disturbance_labels`
  - `disturbance_schedule`
- `utils/plotting_core.py`
  - now resolves output/input/time/disturbance labels from system metadata instead of assuming polymer labels

## Naming normalization

Distillation run outputs now use canonical prefixes so polymer and distillation runs do not collide.

- baseline results are separated by:
  - `mpc_results_nominal.pickle`
  - `mpc_results_disturb_ramp.pickle`
  - `mpc_results_disturb_fluctuation.pickle`
- RL result prefixes now include system, run mode, disturbance profile, and agent/state configuration where needed

## Validation

Validated in `rl-env`:

- Python compile checks for:
  - `systems/distillation/*`
  - modified shared helper/runner/plotting files
- notebook schema validation for all `distillation*_unified.ipynb`
- per-cell Python syntax compilation for all new distillation notebooks
- canonical data copy into `Data/distillation`

Not run in this change:

- live Aspen smoke runs
- full notebook execution against the real distillation plant

Those still require the local Aspen/COM runtime and the configured `.dynf` files.

## Notes

- The archived subtree `DIstillation Column Case/RL_assisted_MPC_DL/` is intentionally left in place as a read-only validation reference.
- The current migration commit is scoped to the canonical main-repo layer and the normalized `Data/distillation` bundle, not the entire archived subtree with its historical run folders.
