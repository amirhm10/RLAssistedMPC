# Distillation Latest Family Runs Analysis

Date: 2026-05-19

## Question

After rerunning the active distillation `.py` workflows, how do the latest results look for:

- weights
- residual
- dueling horizon
- horizon
- Markov correction

The specific follow-up question is whether the newer runs are worse than earlier runs because of network size, discount factor, or reward-shaping changes.

## Files inspected

Latest result bundles:

- `Distillation/Results/distillation_residual_td3_disturb_fluctuation_mismatch_rho_unified/20260519_200250/input_data.pkl`
- `Distillation/Results/distillation_weights_td3_disturb_fluctuation_mismatch_unified/20260519_201951/input_data.pkl`
- `Distillation/Results/distillation_horizon_disturb_fluctuation_mismatch_unified/20260519_202111/input_data.pkl`
- `Distillation/Results/distillation_dueling_horizon_disturb_fluctuation_mismatch_unified/20260519_204534/input_data.pkl`
- `Distillation/Results/distillation_markov_td3_disturb_fluctuation_unified/20260519_210736/input_data.pkl`

Code and defaults:

- `systems/distillation/config.py`
- `systems/distillation/notebook_params.py`
- `distillation_RL_assisted_MPC_weights_unified.py`
- `distillation_RL_assisted_MPC_residual_unified.py`
- `distillation_RL_assisted_MPC_horizons_unified.py`
- `distillation_RL_assisted_MPC_horizons_dueling_unified.py`
- `distillation_RL_assisted_MPC_markov_unified.py`
- `utils/rewards.py`
- `utils/horizon_runner.py`
- `utils/horizon_runner_dueling.py`
- `utils/weights_runner.py`
- `utils/residual_runner.py`
- `utils/markov_runner.py`

Generated analysis assets:

- `report/scripts/analyze_distillation_latest_family_runs_20260519.py`
- `report/figures/distillation_latest_family_runs_20260519/latest_family_summary.csv`
- `report/figures/distillation_latest_family_runs_20260519/history_tail_reward_summary.csv`
- `report/figures/distillation_latest_family_runs_20260519/summary.json`
- `report/figures/distillation_latest_family_runs_20260519/fig_latest_tail_reward_comparison.png`
- `report/figures/distillation_latest_family_runs_20260519/fig_latest_reward_deltas.png`
- `report/figures/distillation_latest_family_runs_20260519/fig_latest_rollout_difference_from_mpc.png`
- `report/figures/distillation_latest_family_runs_20260519/fig_history_tail_rewards_by_family.png`

## Method

The analysis uses each run's saved `input_data.pkl` and compares:

- final average reward
- last-20-subepisode tail average reward
- tail reward relative to the canonical disturbance MPC baseline in `Distillation/Data/mpc_results_disturb_fluctuation.pickle`
- tail reward relative to the best previous saved run in the same method family
- tail reward relative to the best previous run with the same saved agent/algorithm when applicable
- output tracking metrics from the saved `y_rl` and `y_sp`
- input movement from the saved `u_rl`
- high-level config differences against the historical best

The reward being optimized is the relative-band reward:

$$ r_t = \left[-\left(\mathrm{err}_{\mathrm{eff}} + \mathrm{move} + \mathrm{lin}_{\mathrm{out}} + \mathrm{lin}_{\mathrm{in}}\right) + \mathrm{bonus}\right]\cdot \mathrm{reward\_scale}. $$

## Latest result summary

All five latest runs beat the canonical MPC reward baseline, whose last-20-subepisode tail reward is `-0.303` under the current comparison scoring. However, every latest run is below its historical best saved run.

| Family | Latest run | Latest tail reward | Best previous tail reward | Gap to previous best | Gap to MPC baseline |
| --- | --- | ---: | ---: | ---: | ---: |
| Residual TD3 | `20260519_200250` | `13.85` | `18.82` | `-4.98` | `+14.15` |
| Weights TD3 | `20260519_201951` | `10.04` | `20.58` | `-10.54` | `+10.35` |
| Horizon DDQN | `20260519_202111` | `16.92` | `19.97` | `-3.04` | `+17.22` |
| Dueling DDQN | `20260519_204534` | `14.61` | `19.97` | `-5.37` | `+14.91` |
| Markov TD3 | `20260519_210736` | `17.35` | `21.98` | `-4.63` | `+17.65` |

The headline is therefore not that the latest runs failed. They are still better than the canonical MPC reward baseline. The issue is that they are not reproducing the best previous RL outcomes.

## Family-by-family interpretation

### Horizon

The latest horizon run is worse than the previous best by `3.04` tail-reward units.

This one is strongly consistent with reward-shaping changes. The latest run uses:

- `k_rel = [0.3, 0.01]`
- `band_floor_phys = [0.003, 0.2]`
- `Q_diag = [37000, 5000]`

The best previous horizon run, `20260507_214708`, stored:

- `k_rel = [0.3, 0.02]`
- `band_floor_phys = [0.003, 0.3]`
- `Q_diag = [37000, 1500]`

The newer reward is stricter on tray-temperature tracking: the relative band is half as wide, the physical floor is smaller, and the second-output quadratic weight is larger. That can make the same class of closed-loop behavior receive a lower reward and can also change what the DQN learns to prefer.

Network size and gamma are not the likely explanation here. The current horizon defaults remain `hidden_layers = [256, 256, 256]` and `gamma = 0.995`.

### Dueling Horizon

The latest dueling run is worse than the previous best by `5.37` tail-reward units.

This has the same reward-shaping signature as the standard horizon case. The latest dueling run uses the tighter current reward:

- `k_rel = [0.3, 0.01]`
- `band_floor_phys = [0.003, 0.2]`
- `Q_diag = [37000, 5000]`

The best previous dueling run, `20260516_162510`, stored:

- `k_rel = [0.3, 0.02]`
- `band_floor_phys = [0.003, 0.3]`
- `Q_diag = [37000, 1500]`

So the strongest explanation is again reward shaping, not network size or gamma. The current dueling defaults remain `hidden_layers = [256, 256, 256]` and `gamma = 0.995`.

### Weights

The latest weights run looks much worse than the historical best: `10.04` versus `20.58`.

But this comparison is not apples-to-apples. The latest run is TD3, while the best historical weights run was SAC:

- latest: `td3`, `20260519_201951`, tail reward `10.04`
- best previous: `sac`, `20260507_214023`, tail reward `20.58`

When compared only to the previous TD3 weights run, the latest TD3 run is actually much better:

- latest TD3 tail reward: `10.04`
- previous TD3 tail reward: `-0.28`
- same-agent improvement: `+10.33`

So the weights-family degradation relative to "all previous runs" is mostly an algorithm comparison: SAC was the stronger historical weights supervisor. It is not good evidence that TD3 became worse because of network size, gamma, or reward shaping.

The current TD3 weights defaults are:

- actor/critic hidden layers: `[256, 256, 256]`
- `gamma = 0.995`
- `std_start = 0.2`
- `exploration_mode = "param_noise"`
- post-warm-start action freeze: `5` subepisodes
- post-warm-start actor freeze: `5` subepisodes

The latest TD3 weights run also uses mismatch state, while the older TD3 reference was a standard-state run. That further weakens a direct conclusion about network size or gamma.

### Residual

The latest residual TD3 run is worse than the previous best by `4.98` tail-reward units:

- latest: `20260519_200250`, tail reward `13.85`
- previous best: `20260507_212833`, tail reward `18.82`

This is the most ambiguous family. The high-level saved config comparison did not show a major difference between the latest residual run and the May 7 best run. Both are TD3 residual, mismatch-state, rho-authority runs with the same visible release setup.

The latest residual bundle has:

- behavioral cloning enabled
- BC active fraction about `0.10`
- post-warm-start action freeze: `5` subepisodes
- post-warm-start actor freeze: `5` subepisodes

Those features were also present in the recent residual family, so they do not cleanly explain the May 19 drop by themselves.

The residual bundles still do not store `reward_params`, so we cannot prove whether the May 7 and May 19 residual runs used exactly the same reward settings from the result file alone. Given the current code, the reward defaults are the stricter current settings, but the saved residual history is weaker than the horizon/dueling evidence.

Current best interpretation:

- not network size: current TD3 actor/critic remains `[256, 256, 256]`
- not gamma: current TD3 gamma remains `0.995`
- possibly run-to-run RL variability or unlogged reward/config provenance
- possibly the residual authority/BC handoff producing less useful residual corrections in this particular run

### Markov

The latest Markov run is worse than the historical best by `4.63` tail-reward units:

- latest: `20260519_210736`, tail reward `17.35`
- previous best: `20260518_091937`, tail reward `21.98`

This is not primarily a reward-shaping change. Both latest and previous-best Markov bundles stored the same reward signature:

- `k_rel = [0.3, 0.01]`
- `band_floor_phys = [0.003, 0.2]`
- `Q_diag = [37000, 5000]`
- `R_diag = [2500, 2500]`
- `beta = 7`
- `reward_scale = 1`

The important difference is the Markov execution logic:

- latest run: `force_td3_execute = False`
- previous best: `force_td3_execute = True`
- latest run: TD3 priority fallback enabled
- previous best: no saved priority fallback
- latest run: BC disabled
- previous best: BC enabled for 5 subepisodes

The latest Markov run accepted about `95.3%` of candidate actions and fell back about `4.7%` of the time. The previous best was the `td3_only_no_safeguard` variant, which is less conservative and may score better when TD3 proposals happen to be useful. That higher score should be read together with the safety tradeoff: the previous best is not the same controller architecture.

Network size and gamma are not the likely explanation. Current Markov TD3 defaults remain actor/critic `[256, 256, 256]` and `gamma = 0.995`.

## Important logging note

Inside each latest RL bundle, `max |y_rl - y_mpc| = 0` and `max |u_rl - u_mpc| = 0` for the stored bundle-level MPC arrays. However, comparing the same RL trajectories against the canonical MPC baseline file gives nonzero differences.

That means the saved `y_mpc/u_mpc` fields inside each RL bundle appear to mirror the RL trajectory, while the separate canonical MPC pickle is the meaningful baseline. Future reports should continue using `Distillation/Data/mpc_results_disturb_fluctuation.pickle` for MPC comparisons unless this bundle-level logging is cleaned up.

## Answer to the root-cause question

The worse latest results are not mainly explained by network size or gamma.

Evidence:

- All current active families use `gamma = 0.995`.
- DQN/dueling networks remain `[256, 256, 256]`.
- TD3 actor/critic networks remain `[256, 256, 256]`.
- The biggest regressions do not line up with a saved network-size or discount-factor change.

The causes differ by family:

- Horizon and dueling: reward shaping changed and is the strongest explanation.
- Weights: the latest run is TD3, while the historical best is SAC; same-agent TD3 improved over the older TD3 run.
- Residual: no clear high-level config change explains the drop; reward params were not stored, so this needs a rerun with better provenance.
- Markov: the latest run uses guarded/fallback execution, while the historical best was a less conservative TD3-only/no-safeguard variant.

## Recommendations

1. Restore reward provenance in every saved bundle.
   Weights and residual should store `reward_params`, just like horizon, dueling, and Markov.

2. Run a controlled reward ablation for horizon and dueling.
   Re-run latest code once with the older reward profile: `k_rel = [0.3, 0.02]`, `band_floor_phys = [0.003, 0.3]`, `Q_diag = [37000, 1500]`.

3. Compare weights as SAC-vs-SAC and TD3-vs-TD3.
   The latest TD3 run should not be judged against the best SAC run when diagnosing TD3 behavior.

4. For residual, repeat the May 19 run with an explicit seed and full provenance.
   Save reward params, TD3 hyperparameters, BC schedule, authority traces, and action-source traces.

5. For Markov, compare guarded and no-safeguard variants separately.
   The no-safeguard variant may score better but is a different safety regime.

6. Fix or relabel bundle-level `y_mpc/u_mpc`.
   Right now those arrays mirror the RL trajectory in the latest bundles, so canonical MPC comparisons must use the separate MPC pickle.
