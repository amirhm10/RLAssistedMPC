# 2026-05-11 Distillation Matrix Latest Follow-up

## Context

Reviewed the newest scalar distillation matrix TD3 disturbance run:

- `Distillation/Results/distillation_matrix_td3_disturb_fluctuation_mismatch_unified/20260511_183650`

The user noticed that the final episode looked mixed:

- second setpoint block seemed much better
- first setpoint block looked jittery

## What we found

- The saved RL bundle’s internal `y_mpc` and `u_mpc` are duplicates of the RL trajectories because of the current `utils/plotting_core.py` storage helper.
- The correct disturbance MPC reference for analysis is `Distillation/Data/mpc_results_disturb_fluctuation.pickle`.
- Under the real baseline, the newest scalar matrix run is still strongly negative overall.
- The final episode does split sharply by setpoint block:
  - block 1 is highly unstable with enormous input movement and high alpha variation
  - block 2 is much calmer
- However, the calmer second block is still not better than disturbance MPC on tracking metrics.
- The same first-block instability already appears across the last 20 episodes, so it is not just a one-off final-test artifact.

## Assets added

- `report/scripts/generate_distillation_matrix_latest_followup_assets_20260511.py`
- `report/figures/distillation_matrix_latest_followup_20260511/`
- report update in `report/distillation_matrix_structured_step4g_latest_2026_05_04.md`

## Interpretation

The latest scalar matrix result does not support a success claim. It does suggest a more specific mechanism:

- the fully released scalar multiplier policy is especially unstable in the first setpoint block
- later in the episode the multiplier dynamics collapse toward a calmer regime
- that later calm can look visually encouraging, but it still does not beat disturbance MPC quantitatively
