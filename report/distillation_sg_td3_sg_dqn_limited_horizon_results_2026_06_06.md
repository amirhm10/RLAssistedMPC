# Distillation SG-TD3 and Limited-Horizon SG-DQN Results

Generated on 2026-06-06 from saved result bundles only; Aspen was not relaunched.

## Executive Takeaways

- Best tail performance is **Residual SG-TD3 current** with tail reward 31.171, which is 24.780 above OF-MPC.
- The most stable current run by post-warm failures is **Weights SG-TD3 current**: 0 negative post-warm episodes and tail reward 19.046.
- Limited-grid SG-DQN improved over the old wide-grid SG-DQN by 1.414 reward points, reduced T85 MAE by 0.008, and removed the previous negative post-warm episode.
- The limited horizon grid is not just smaller; the saved recipes are triangular: `Np = 6..11` and `Nc = 3..Np`, giving 39 valid actions rather than a full Cartesian 54-action grid.
- No saved limited-grid dueling-DQN mismatch bundle was found under `Distillation/Results`; the current dueling row is therefore the older wide-grid June 4 context run only.

## Method Snapshot

All reported rewards are recomputed from the saved trajectories with the current distillation reward, so runs with different logged reward revisions are compared on the same scale.

$$ r_t = -e_t^\top Q e_t - \Delta u_t^\top R \Delta u_t + b(e_t, y_{\mathrm{sp},t}), $$

where the report uses the saved scaled-deviation output errors, scaled input moves, and physical setpoints to reconstruct the same band-gated bonus term used by the current distillation reward code.

SG-TD3 weights, residual, and Markov runs use continuous actors with twin critics and a supervisor gate. The current runs keep fixed MPC horizons `(6, 3)`. Horizon SG-DQN is discrete: the agent selects an `(Np, Nc)` recipe and the supervisor gate compares the learned Q score for the candidate action against the OF-MPC/default supervisor action. The dueling DQN variant changes the Q-network decomposition, not the fact that the gate is still single-Q discrete.

## Current Run Summary

| Run | Tail reward | Delta vs OF-MPC | Final reward | Worst post-warm | Neg post-warm | T85 MAE | x24 MAE | Policy step frac tail | Policy step frac first live |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OF-MPC | 6.391 | 0.000 | 6.926 | 4.488 | 0 | 0.192 | 0.00155 |  |  |
| Weights SG-TD3 current | 19.046 | 12.655 | 18.451 | 7.460 | 0 | 0.156 | 0.00096 | 0.714 | 0.294 |
| Residual SG-TD3 current | 31.171 | 24.780 | 33.116 | -15.997 | 5 | 0.070 | 0.00105 | 0.520 | 0.312 |
| Markov SG-TD3 current param-noise | 27.348 | 20.957 | 31.217 | -1.972 | 2 | 0.062 | 0.00115 | 0.366 | 0.037 |
| Horizon SG-DQN limited 39 | 12.473 | 6.082 | 11.677 | 7.024 | 0 | 0.182 | 0.00098 | 0.219 | 0.057 |

## Change Against Previous Mismatch References

| Family | Tail reward change | T85 MAE change | x24 MAE change | Neg post-warm change | Tail policy step change | First-live policy step change |
|---|---:|---:|---:|---:|---:|---:|
| weights | 0.634 | 0.001 | -0.00026 | 0 | 0.135 | -0.039 |
| residual | 2.256 | -0.003 | -0.00009 | 4 | 0.096 | 0.296 |
| markov | 0.725 | -0.005 | 0.00004 | 1 | -0.015 | 0.013 |
| horizon | 1.414 | -0.008 | 0.00004 | -1 | -0.014 | -0.007 |

Positive reward change is good; negative T85 or x24 MAE change is good.

## Family Interpretation

### Weights SG-TD3

The weights supervisor is the safest successful continuous run. Tail reward rose from 18.412 to 19.046, with zero negative post-warm episodes in both runs. The tail gate became more policy-forward, increasing from 0.579 to 0.714, while the first-live gate stayed conservative (0.294).

### Residual SG-TD3

Residual is the strongest tail performer but not the cleanest handover. Tail reward improved from 28.916 to 31.171, and T85 MAE improved to 0.070. The cost is fragility: negative post-warm episodes increased from 1 to 5, with the worst collapse at -15.997. This matches the residual-action risk we expected: once accepted, a residual move can directly perturb the plant input rather than only reshaping the MPC objective.

### Markov SG-TD3

Markov improved modestly over the previous Markov reference: tail reward 26.622 to 27.348, T85 MAE 0.067 to 0.062. It remains conservative at release, with first-live policy step fraction only 0.037. The saved bundle reports top-level `state_mode='standard'`, but the Markov-specific fields are `markov_state_mode='mismatch'` and `markov_agent_state_features='markov'`; therefore the run should be interpreted as the Markov/mismatch-conditioned param-noise run, with a generic top-level logging ambiguity.

### Horizon SG-DQN

The limited 39-action horizon DQN is a real improvement over the wide 87-action SG-DQN: tail reward 11.058 to 12.473, T85 MAE 0.191 to 0.182, and post-warm negative episodes 1 to 0. The heatmap still shows a diffuse policy: all 39 recipes appear in the tail, and the most common pair only occupies 0.144 of tail steps. So the narrowed range helped stability, but did not yet create a confident horizon policy.

## Failure Diagnostics

The worst current post-warm episodes show two different failure modes. Residual has a real handover shock; Markov has a later disturbance-region dip while the gate is still mostly supervisor-led; weights and limited SG-DQN do not collapse below zero.

| Run | Worst subepisode | Reward | T85 MAE | x24 MAE | Policy step frac | Median gate advantage |
|---|---:|---:|---:|---:|---:|---:|
| Weights SG-TD3 current | 16 | 7.460 | 0.183 | 0.00156 | 0.182 | -3.554 |
| Residual SG-TD3 current | 18 | -15.997 | 0.195 | 0.00571 | 0.403 | -2.713 |
| Markov SG-TD3 current param-noise | 54 | -1.972 | 0.238 | 0.00149 | 0.120 | -140.226 |
| Horizon SG-DQN limited 39 | 35 | 7.024 | 0.187 | 0.00119 | 0.068 | 0.000 |

This is why the residual tail result should not be read as universally safer than weights: residual eventually learns the best tail behavior, but its first live window accepts enough direct input residuals to create the worst short-term collapse. Markov is more conservative at release, so its weak episodes look more like insufficient correction around a harder disturbance region than a gate handover failure.

## Horizon Candidate Diagnostics

| Run | Tail top pairs |
|---|---|
| Horizon SG-DQN limited 39 | (6, 4) at 14.4 percent; (6, 6) at 13.1 percent; (6, 3) at 12.5 percent; (8, 6) at 7.0 percent; (6, 5) at 4.2 percent |
| Horizon SG-DQN wide 87 | (6, 3) at 6.8 percent; (12, 11) at 5.0 percent; (5, 2) at 3.6 percent; (14, 4) at 2.7 percent; (14, 8) at 2.5 percent |
| Dueling SG-DQN wide 87 | (6, 3) at 28.3 percent; (11, 11) at 19.0 percent; (12, 9) at 7.8 percent; (4, 2) at 5.8 percent; (9, 6) at 5.5 percent |

The limited run shifts mass toward short prediction horizons, especially `(6, 4)`, `(6, 6)`, and `(6, 3)`. The old wide SG-DQN spent tail probability on both very short and very long pairs, while the old dueling run concentrated on `(6, 3)` and `(11, 11)` but still had poor tracking. That argues for a reduced candidate set, not for returning to the full wide grid.

## Figures

- [fig_current_tail_reward_tracking.png](report/figures/distillation_sg_limited_horizons_20260606/fig_current_tail_reward_tracking.png)
- [fig_current_vs_previous_delta.png](report/figures/distillation_sg_limited_horizons_20260606/fig_current_vs_previous_delta.png)
- [fig_current_episode_rewards.png](report/figures/distillation_sg_limited_horizons_20260606/fig_current_episode_rewards.png)
- [fig_current_gate_policy_fraction.png](report/figures/distillation_sg_limited_horizons_20260606/fig_current_gate_policy_fraction.png)
- [fig_horizon_tail_heatmaps.png](report/figures/distillation_sg_limited_horizons_20260606/fig_horizon_tail_heatmaps.png)

## Recommended Next Recipes

For the next horizon-only distillation pass, keep the reduced lower bound and avoid returning to the full 87-action grid. The current data support `Np = 6..11` and `Nc = 3..Np` as a better candidate set than the old wide grid. A slightly more focused follow-up is worth testing: `Np = 6..10`, `Nc = 3..min(Np, 7)`, while keeping `(6, 3)` as the supervisor/default action. That keeps the pairs used most often by the successful limited run and removes some high-control-horizon actions that still appear exploratory rather than decisively useful.

For continuous supervisors, the current ranking is:

- Residual SG-TD3 is best for final/tail tracking but needs acceptance judged with the negative-episode risk visible.
- Weights SG-TD3 is the cleanest stable candidate and the easiest to defend as a robust improvement.
- Markov SG-TD3 is useful and conservative, but its state-mode logging should be cleaned before using the bundle as final paper evidence.
- Horizon SG-DQN limited is improved but still behind the continuous supervisors.

## Provenance

Files inspected:

- `report/distillation_standard_mode_sg_analysis_2026_06_05.md`
- `report/scripts/analyze_distillation_standard_mode_sg_20260605.py`
- `report/scripts/analyze_distillation_dueling_horizon_history_20260604.py`
- `report/scripts/analyze_distillation_sg_dqn_horizon_followup_20260604.py`
- `distillation_RL_assisted_MPC_horizons_supervisor_gated_dqn_unified.py`
- `distillation_RL_assisted_MPC_horizons_supervisor_gated_dueling_dqn_unified.py`
- `distillation_RL_assisted_MPC_weights_supervisor_gated_td3_critic_warm_unified.py`
- `distillation_RL_assisted_MPC_residual_supervisor_gated_td3_critic_warm_unified.py`
- `distillation_RL_assisted_MPC_markov_supervisor_gated_td3_critic_warm_unified.py`
- `tests/test_supervisor_gated_horizon_runners.py`
- `Distillation/Data/mpc_results_disturb_fluctuation.pickle`
- `Distillation/Results/distillation_weights_sg_td3_critic_warm3_margin0_sup001_gauss015_003_manual_off_disturb_fluctuation_mismatch/20260606_075201/input_data.pkl`
- `Distillation/Results/distillation_weights_sg_td3_critic_warm3_margin0_sup001_gauss015_003_manual_off_disturb_fluctuation_mismatch/20260603_124214/input_data.pkl`
- `Distillation/Results/distillation_residual_sg_td3_critic_warm3_manual_off_disturb_fluctuation_mismatch_no_rho/20260606_074932/input_data.pkl`
- `Distillation/Results/distillation_residual_sg_td3_critic_warm3_manual_off_disturb_fluctuation_mismatch_no_rho/20260602_125954/input_data.pkl`
- `Distillation/Results/distillation_markov_sg_td3_critic_warm3_ls_else_mpc_shadow_disturb_fluctuation_mismatch_paramnoise/20260606_105404/input_data.pkl`
- `Distillation/Results/distillation_markov_sg_td3_critic_warm3_ls_else_mpc_shadow_disturb_fluctuation_unified/20260602_192543/input_data.pkl`
- `Distillation/Results/distillation_horizon_sg_dqn_critic_warm3_default_ofmpc_eps02_002_disturb_fluctuation_mismatch_np6_11_nc3_11/20260606_071451/input_data.pkl`
- `Distillation/Results/distillation_horizon_sg_dqn_critic_warm3_default_ofmpc_eps02_002_disturb_fluctuation_mismatch/20260604_103231/input_data.pkl`
- `Distillation/Results/distillation_dueling_horizon_sg_dqn_critic_warm3_default_ofmpc_eps02_002_disturb_fluctuation_mismatch/20260604_105140/input_data.pkl`

Generated outputs:

- `report/figures/distillation_sg_limited_horizons_20260606/distillation_sg_limited_summary.csv`
- `report/figures/distillation_sg_limited_horizons_20260606/distillation_sg_limited_summary.json`
- `report/figures/distillation_sg_limited_horizons_20260606/distillation_sg_limited_episode_diagnostics.csv`
- `report/figures/distillation_sg_limited_horizons_20260606/distillation_sg_limited_horizon_pairs.csv`
- `report/figures/distillation_sg_limited_horizons_20260606/fig_current_tail_reward_tracking.png`
- `report/figures/distillation_sg_limited_horizons_20260606/fig_current_vs_previous_delta.png`
- `report/figures/distillation_sg_limited_horizons_20260606/fig_current_episode_rewards.png`
- `report/figures/distillation_sg_limited_horizons_20260606/fig_current_gate_policy_fraction.png`
- `report/figures/distillation_sg_limited_horizons_20260606/fig_horizon_tail_heatmaps.png`
