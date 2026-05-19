# Distillation Run History Audit

Date: 2026-05-19

## Question

Why do some older distillation runs for horizon, dueling horizon, weights, and residual supervision look better than the more recent runs? Is that mainly reward shaping, or did the code path change?

## Files inspected

Code and defaults:

- `utils/rewards.py`
- `systems/distillation/config.py`
- `systems/distillation/notebook_params.py`
- `distillation_RL_assisted_MPC_horizons_unified.py`
- `distillation_RL_assisted_MPC_horizons_dueling_unified.py`
- `distillation_RL_assisted_MPC_weights_unified.py`
- `distillation_RL_assisted_MPC_residual_unified.py`
- `distillation_RL_assisted_MPC_markov_unified.py`
- `report/distillation_reward_audit.md`

Saved results and analysis assets:

- `Distillation/Results/distillation_horizon_disturb_fluctuation_standard_unified/*/input_data.pkl`
- `Distillation/Results/distillation_horizon_disturb_fluctuation_mismatch_unified/*/input_data.pkl`
- `Distillation/Results/distillation_dueling_horizon_disturb_fluctuation_standard_unified/*/input_data.pkl`
- `Distillation/Results/distillation_dueling_horizon_disturb_fluctuation_mismatch_unified/*/input_data.pkl`
- `Distillation/Results/distillation_weights_td3_disturb_fluctuation_standard_unified/*/input_data.pkl`
- `Distillation/Results/distillation_weights_sac_disturb_fluctuation_standard_unified/*/input_data.pkl`
- `Distillation/Results/distillation_weights_sac_disturb_fluctuation_mismatch_unified/*/input_data.pkl`
- `Distillation/Results/distillation_residual_td3_disturb_fluctuation_mismatch_rho_unified/*/input_data.pkl`
- `Distillation/Results/distillation_residual_sac_disturb_fluctuation_mismatch_rho_unified/*/input_data.pkl`
- `report/figures/distillation_run_history_audit_20260519/representative_run_summary.csv`
- `report/figures/distillation_run_history_audit_20260519/summary.json`
- `report/figures/distillation_run_history_audit_20260519/fig_reward_and_identity_audit.png`

## Reward structure

The current unified distillation notebooks all build the reward through `utils/rewards.py`:

$$
r_t = \left[-\left(\mathrm{err}_{\mathrm{eff}} + \mathrm{move} + \mathrm{lin}_{\mathrm{out}} + \mathrm{lin}_{\mathrm{in}}\right) + \mathrm{bonus}\right]\cdot \mathrm{reward\_scale}.
$$

The reward is therefore very sensitive to the relative band settings `k_rel` and `band_floor_phys`, because those parameters determine when the inside-band bonus turns on and how aggressively tracking error is treated as being outside the acceptable band.

## Main findings

### 1. Reward shaping did change

For the saved horizon and dueling-horizon distillation bundles, the reward parameters stored inside the run bundles changed between the older stronger-looking runs and the latest runs.

Representative examples:

| Method | Older run | Latest run | Older reward signature | Latest reward signature |
| --- | --- | --- | --- | --- |
| Horizon | `20260416_192434` | `20260518_141636` | `k_rel=[0.3, 0.02]`, `band_floor=[0.003, 0.3]`, `beta=7`, `scale=1.0` | `k_rel=[0.3, 0.01]`, `band_floor=[0.003, 0.2]`, `beta=7`, `scale=1.0` |
| Dueling | `20260420_173113` | `20260518_140746` | `k_rel=[0.3, 0.02]`, `band_floor=[0.003, 0.3]`, `beta=7`, `scale=1.0` | `k_rel=[0.3, 0.01]`, `band_floor=[0.003, 0.2]`, `beta=7`, `scale=1.0` |

This is a real change, not just visual noise:

- the second-output relative tolerance tightened from `0.02` to `0.01`
- the second-output physical floor tightened from `0.3` to `0.2`

That makes the reward harsher on tray-temperature error near the target region, so newer reward curves can look worse even if the underlying trajectory is not worse.

The current repo-wide distillation reward default in `systems/distillation/config.py` is the tighter version:

- `k_rel = [0.3, 0.01]`
- `band_floor_phys = [0.003, 0.2]`
- `Q_diag = [3.7e4, 5.0e3]`
- `R_diag = [2.5e3, 2.5e3]`
- `reward_scale = 1.0`

That also differs from the older archived distillation RL reward family documented in `report/distillation_reward_audit.md`, which used:

- `k_rel = [0.3, 0.02]`
- `band_floor_phys = [0.003, 0.3]`
- `Q_diag = [3.7e4, 1.5e3]`

So yes, reward shaping is one real reason the older runs can appear better.

### 2. The code path also changed beyond reward

The newer runs are not only using a different reward profile. They are also running under different state and training logic.

Observed differences in the stored run configs:

- Horizon:
  - older stronger-looking run: `state_mode = "standard"`
  - latest run: `state_mode = "mismatch"`
  - latest run also stores `observer_update_alignment = "legacy_previous_measurement"`

- Dueling horizon:
  - older stronger-looking run: `state_mode = "standard"`
  - latest run: `state_mode = "mismatch"`
  - latest run also stores `observer_update_alignment = "legacy_previous_measurement"`

- Weights:
  - older representative SAC run: `state_mode = "standard"`
  - latest SAC run: `state_mode = "mismatch"`
  - latest run adds `post_warm_start_action_freeze_subepisodes = 5`
  - latest run adds `post_warm_start_actor_freeze_subepisodes = 5`

- Residual:
  - both representative runs use mismatch state with `rho`
  - latest run adds `observer_update_alignment = "legacy_previous_measurement"`
  - latest run adds `post_warm_start_action_freeze_subepisodes = 5`
  - latest run adds `post_warm_start_actor_freeze_subepisodes = 5`
  - latest run enables behavioral cloning, while the older representative run did not

So the comparison is not "same algorithm, same state, same reward, different luck." The newer runs are scientifically different runs.

### 3. The strongest result: the stored RL and MPC trajectories are identical

This was the most important finding from the saved bundles.

Across the scanned distillation result folders for:

- horizon
- dueling horizon
- weights
- residual

I checked `34` saved run bundles. In all `34/34`:

- `max |y_rl - y_mpc| = 0`
- `max |u_rl - u_mpc| = 0`

This means that in the stored arrays, the RL trajectory and the baseline MPC trajectory are exactly the same for every scanned run.

Implication:

- the older runs are not demonstrably "much better" in closed-loop output tracking inside these saved bundles
- what changed most visibly is the reward trace, metadata, and internal action logs
- there may also be a logging or serialization issue where the saved `y_rl/u_rl` arrays are being overwritten by or aliased to the MPC trajectory

This finding matters more than the reward change, because it means the saved output trajectories do not currently support the claim that one distillation RL run tracked better than another.

### 4. Reward provenance is incomplete in some families

The horizon and dueling bundles store `reward_params` directly. The weights and residual bundles I inspected do not.

That does not mean those families used the same reward forever. It means the saved bundle does not preserve enough reward provenance to prove it after the fact.

Because the current code routes these families through the same centralized distillation reward defaults, it is still plausible that reward changes affected weights and residual too. But for those two families, the saved bundle evidence is weaker than for horizon and dueling.

## Representative summary

From `report/figures/distillation_run_history_audit_20260519/representative_run_summary.csv`:

| Method | Older tail reward | Latest tail reward | Main code-path difference |
| --- | --- | --- | --- |
| Horizon | `17.87` | `15.61` | `standard -> mismatch`, tighter reward band on output 2 |
| Dueling | `19.87` | `15.86` | `standard -> mismatch`, tighter reward band on output 2 |
| Weights | `19.30` | `15.88` | `standard -> mismatch`, added 5/5 post-warm-start freeze |
| Residual | `18.10` | `12.14` | added observer alignment, 5/5 freeze, behavioral cloning |

These reward drops are real, but they are not accompanied by nonzero stored RL-vs-MPC output differences in the saved arrays.

## Interpretation

My current best explanation is:

1. The older runs often look better partly because the reward really was more permissive, especially for horizon-family distillation runs.
2. The newer runs also changed state representation and training scaffolding, so they are not directly apples-to-apples.
3. The saved bundle evidence does not show better closed-loop RL tracking in the older runs, because the stored RL and MPC trajectories are identical across all scanned runs.

So if you were reacting mainly to reward curves or summary panels, then yes, reward shaping is a major part of the story.

If you were reacting to the actual output plots, then the stronger concern is that the saved result path is not preserving a distinct RL trajectory, or the compare plotting path is effectively showing MPC-equivalent traces for both sides.

## Recommended next checks

1. Save explicit executed assisted-controller traces in every family.
   At minimum: mapped action, executed action, assisted MPC settings, `u_rl`, `u_mpc`, `y_rl`, `y_mpc`, and a boolean that confirms whether the assisted policy was actually released.

2. Add `reward_params` to the saved bundles for weights and residual.
   Right now those bundles are weaker than horizon/dueling for historical auditing.

3. Do one short rerun per family and assert online that RL differs from MPC after warm start.
   A simple sanity check is:
   - after release, `np.max(np.abs(u_rl - u_mpc)) > 0`
   - and for horizon/matrix/weights/residual, log the exact applied supervisory action each step

4. Audit the plotting path that writes `y_rl` and `u_rl`.
   The current saved results strongly suggest either:
   - RL is not materially changing the closed-loop trajectory in these runs, or
   - the saved RL arrays are not preserving the actual assisted rollout

## Output files created for this audit

- `report/distillation_run_history_audit_2026_05_19.md`
- `report/scripts/analyze_distillation_run_history_20260519.py`
- `report/figures/distillation_run_history_audit_20260519/representative_run_summary.csv`
- `report/figures/distillation_run_history_audit_20260519/summary.json`
- `report/figures/distillation_run_history_audit_20260519/fig_reward_and_identity_audit.png`
