# Method-Scoped Conditioning Change Log

Date: 2026-04-20

This note tracks the two related changesets applied to the unified polymer and distillation workflows:

1. method-scoped conditioning and residual refresh
2. notebook-default activation of those new behaviors

## Scope

The work applies to the active unified RL-assisted MPC notebooks and shared runners for both:

- polymer CSTR
- distillation column

The baseline MPC notebooks remain baseline-only, but their default `run_mode` stays on disturbance as before.

## Change Set 1: Shared Runtime Implementation

The shared runner/state pipeline was extended so the new behavior exists behind config flags:

- grouped running normalization for physical `xhat`
- configurable mismatch feature transforms
- current-measurement observer-alignment option
- residual-family-only `rho` redesign hooks
- residual deadband hooks near setpoint
- raw and transformed mismatch logging for diagnostics

Primary implementation surface:

- [utils/observation_conditioning.py](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/utils/observation_conditioning.py)
- [utils/state_features.py](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/utils/state_features.py)
- [utils/residual_authority.py](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/utils/residual_authority.py)
- shared mismatch-capable runners under [utils/](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/utils)

Related report artifacts:

- [rl_state_scaling_diagnostics.md](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/report/rl_state_scaling_diagnostics.md)
- [generate_rl_state_scaling_report.py](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/report/generate_rl_state_scaling_report.py)

## Change Set 2: New Notebook Defaults Activated

The new runtime flags are now the default notebook-facing behavior for the RL notebooks, so the unified notebooks run directly in the refreshed path without manual config edits.

### Polymer defaults

All active polymer RL notebook families now default to:

- `run_mode="disturb"`
- `state_mode="mismatch"` for single-agent RL families
- `horizon_state_mode="mismatch"`
- `matrix_state_mode="mismatch"`
- `weights_state_mode="mismatch"`
- `residual_state_mode="mismatch"` for combined

Mismatch-conditioning defaults now resolve to:

- `base_state_norm_mode="running_zscore_physical_xhat"`
- `mismatch_feature_transform_mode="signed_log"`
- `observer_update_alignment="current_measurement_corrector"`

Residual-family defaults now resolve to:

- `rho_mapping_mode="exp_raw_tracking"`
- `residual_zero_deadband_enabled=True`

Source of truth:

- [systems/polymer/notebook_params.py](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/systems/polymer/notebook_params.py)

### Distillation defaults

All active distillation RL notebook families now default to:

- `run_mode="disturb"`
- `disturbance_profile="fluctuation"`
- `state_mode="mismatch"` for single-agent RL families
- `horizon_state_mode="mismatch"`
- `matrix_state_mode="mismatch"`
- `weights_state_mode="mismatch"`
- `residual_state_mode="mismatch"` for combined

Mismatch-conditioning defaults now resolve to:

- `base_state_norm_mode="fixed_minmax"`
- `mismatch_feature_transform_mode="signed_log"`
- `observer_update_alignment="current_measurement_corrector"`

Residual-family defaults now resolve to:

- `rho_mapping_mode="exp_raw_tracking"`
- `residual_zero_deadband_enabled=True`

Source of truth:

- [systems/distillation/notebook_params.py](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/systems/distillation/notebook_params.py)

## Notebook Entry Points Updated

The unified RL notebooks were updated so they now forward the new state-conditioning and residual-authority fields into the shared runner configs instead of silently staying on legacy behavior.

Polymer notebooks updated:

- [RL_assisted_MPC_horizons_unified.ipynb](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/RL_assisted_MPC_horizons_unified.ipynb)
- [RL_assisted_MPC_horizons_dueling_unified.ipynb](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/RL_assisted_MPC_horizons_dueling_unified.ipynb)
- [RL_assisted_MPC_matrices_unified.ipynb](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/RL_assisted_MPC_matrices_unified.ipynb)
- [RL_assisted_MPC_structured_matrices_unified.ipynb](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/RL_assisted_MPC_structured_matrices_unified.ipynb)
- [RL_assisted_MPC_weights_unified.ipynb](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/RL_assisted_MPC_weights_unified.ipynb)
- [RL_assisted_MPC_residual_unified.ipynb](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/RL_assisted_MPC_residual_unified.ipynb)
- [RL_assisted_MPC_reidentification_unified.ipynb](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/RL_assisted_MPC_reidentification_unified.ipynb)
- [RL_assisted_MPC_combined_unified.ipynb](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/RL_assisted_MPC_combined_unified.ipynb)

Distillation notebooks updated:

- [distillation_RL_assisted_MPC_horizons_unified.ipynb](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/distillation_RL_assisted_MPC_horizons_unified.ipynb)
- [distillation_RL_assisted_MPC_horizons_dueling_unified.ipynb](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/distillation_RL_assisted_MPC_horizons_dueling_unified.ipynb)
- [distillation_RL_assisted_MPC_matrices_unified.ipynb](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/distillation_RL_assisted_MPC_matrices_unified.ipynb)
- [distillation_RL_assisted_MPC_structured_matrices_unified.ipynb](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/distillation_RL_assisted_MPC_structured_matrices_unified.ipynb)
- [distillation_RL_assisted_MPC_weights_unified.ipynb](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/distillation_RL_assisted_MPC_weights_unified.ipynb)
- [distillation_RL_assisted_MPC_residual_unified.ipynb](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/distillation_RL_assisted_MPC_residual_unified.ipynb)
- [distillation_RL_assisted_MPC_reidentification_unified.ipynb](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/distillation_RL_assisted_MPC_reidentification_unified.ipynb)
- [distillation_RL_assisted_MPC_combined_unified.ipynb](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/distillation_RL_assisted_MPC_combined_unified.ipynb)

## Practical Effect

If you open the unified RL notebooks now and run them without editing the config cells first, they will start from the new disturbance + mismatch defaults and will pass the refreshed conditioning/residual settings down to the shared runners.

That means:

- polymer mismatch notebooks now exercise running `xhat` normalization by default
- both systems use signed-log mismatch transforms by default
- both systems use current-measurement observer alignment by default where the shared mismatch path reads it from the controller config
- residual and combined residual branches now default to the new `rho` map and deadband settings

## How To Return To Legacy Behavior

If a specific experiment should reproduce the older path, change the notebook defaults or override the notebook cell values back to:

- `state_mode="standard"` where needed
- `base_state_norm_mode="fixed_minmax"`
- `mismatch_feature_transform_mode="hard_clip"`
- `observer_update_alignment="legacy_previous_measurement"`
- `rho_mapping_mode="clipped_linear"`
- `residual_zero_deadband_enabled=False`

## Validation Completed

Validation for this refresh:

- imported polymer and distillation notebook defaults directly
- confirmed every active RL family resolves to disturbance + mismatch defaults
- confirmed residual defaults resolve to `exp_raw_tracking` plus deadband
- re-read every edited notebook with `nbformat`
- confirmed the new config keys are present in the notebook code cells
