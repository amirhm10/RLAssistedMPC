# Distillation Column Transfer Approval Summary

## Recheck Result

I rechecked the updated distillation notebooks you changed.

The earlier warning about stale polymer/CSTR content is no longer correct for the main source notebooks.

### Now confirmed

- `[systemIdentification.ipynb](c:\Users\HAMEDI\Desktop\RL_assisted_MPC_polymer\DIstillation Column Case\RL_assisted_MPC_DL\systemIdentification.ipynb)` is now distillation-based.
- It writes:
  - `system_dict_new.pickle`
  - `scaling_factor_new.pickle`
- The baseline MPC notebooks are now:
  - `[MPCOffsetFree.ipynb](c:\Users\HAMEDI\Desktop\RL_assisted_MPC_polymer\DIstillation Column Case\RL_assisted_MPC_DL\MPCOffsetFree.ipynb)`
  - `[MPCOffsetFreeDistRamp.ipynb](c:\Users\HAMEDI\Desktop\RL_assisted_MPC_polymer\DIstillation Column Case\RL_assisted_MPC_DL\MPCOffsetFreeDistRamp.ipynb)`
  - `[MPCOffsetFreeDistRampNoise.ipynb](c:\Users\HAMEDI\Desktop\RL_assisted_MPC_polymer\DIstillation Column Case\RL_assisted_MPC_DL\MPCOffsetFreeDistRampNoise.ipynb)`
- Distillation runtime is Aspen-based through `DistillationColumnAspen`.

## Main Remaining Issues

- file naming is still inconsistent:
  - `system_dict` vs `system_dict_new`
  - `scaling_factor` vs `scaling_factor_new`
  - `mpc_results_200`
  - `mpc_results_dist_ramp`
  - `mpc_result_dist_ramp_noise`
  - `mpc_results_dist_ramp_noise_200`
- plot labels need to be moved into system metadata instead of staying notebook-specific
- I still did not find a native distillation residual notebook family

That last point does not stop migration anymore, because residual can be ported from the existing shared unified residual modules in the main repo.

## Revised Transfer Scope

Recommended target notebooks:

- `distillation_MPCOffsetFree_unified.ipynb`
- `distillation_RL_assisted_MPC_horizons_unified.ipynb`
- `distillation_RL_assisted_MPC_matrices_unified.ipynb`
- `distillation_RL_assisted_MPC_weights_unified.ipynb`
- `distillation_RL_assisted_MPC_residual_unified.ipynb`
- `distillation_RL_assisted_MPC_combined_unified.ipynb`

## Recommended Directory Layout

- `[systems/distillation](c:\Users\HAMEDI\Desktop\RL_assisted_MPC_polymer\systems\distillation)`
- `[Data/distillation](c:\Users\HAMEDI\Desktop\RL_assisted_MPC_polymer\Data\distillation)`
- `[Result/distillation](c:\Users\HAMEDI\Desktop\RL_assisted_MPC_polymer\Result\distillation)`

## Recommended Canonical Naming

Baseline MPC outputs:

- `mpc_results_nominal.pickle`
- `mpc_results_ramp.pickle`
- `mpc_results_fluctuation.pickle`

System/model data:

- `system_dict.pickle`
- `scaling_factor.pickle`
- `min_max_states.pickle`

## Decisions To Approve

Please confirm these before implementation:

1. Use `systems/distillation` + `Data/distillation` + `Result/distillation`.
2. Keep shared runners and plotting, but create separate distillation unified notebooks.
3. Include residual in the distillation migration scope.
4. Use the updated distillation system-ID notebook as a valid source notebook.
5. Normalize file names during migration instead of preserving the current mixed naming.

## Detailed Document

Full analysis is here:

- `[distillation_column_transfer_extensive.md](c:\Users\HAMEDI\Desktop\RL_assisted_MPC_polymer\report\distillation_column_transfer_extensive.md)`
