## June 7 Replay-40k Warm-3 Setup

Generated on 2026-06-07 from saved result bundles only; Aspen was not relaunched.

### Setup Being Tested

This pass keeps the mismatch-state distillation SG runners, shortens the critic-only handoff to warm-3, and uses the distillation replay default of 40,000 transitions. Residual and Markov SG-TD3 now use parameter noise starting at 0.10 and ending at 0.02, with `advantage_margin = 0.5`; weights SG-TD3 keeps Gaussian exploration and `advantage_margin = 0.0`. Both horizon runners use the limited triangular grid `Np = 6..11`, `Nc = 3..Np`.

The continuous SG-TD3 gate can be summarized as accepting the actor only when the conservative policy score beats the supervisor score by the configured margin; otherwise the supervisor action is executed. For residual, the supervisor action is zero residual; for weights it is the identity multiplier; for Markov it is the LS-or-MPC Markov correction. SG-DQN performs the same handoff idea over discrete horizon recipes using one learned Q score.

### Executive Takeaways

- Best tail performance in this setup is **Markov SG-TD3 margin0.5 replay40k** with tail reward 29.931, 23.540 above OF-MPC.
- The cleanest post-warm stability is **Weights SG-TD3 replay40k**, with 0 negative post-warm episodes and tail reward 15.279.
- Residual parameter noise plus margin 0.5 changed the character of the run: tail reward moved from 31.171 to 29.801, while negative post-warm episodes moved from 5 to 1.
- Markov margin 0.5 and softer parameter noise changed tail reward from 27.348 to 29.931; the key diagnostic is whether its poorer episodes are from conservative under-correction or from the actor being accepted too often.
- Dueling SG-DQN now has a saved limited-grid mismatch run. Compared with the old wide-grid dueling reference, tail reward changed by 5.014 and negative post-warm episodes changed by -43.

### June 7 Current Run Summary

| Run | Timestamp | Tail reward | Delta vs OF-MPC | Final reward | Worst post-warm | Neg post-warm | T85 MAE | x24 MAE | Tail policy frac | First-live policy frac |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OF-MPC | Data | 6.391 | 0.000 | 6.926 | 4.488 | 0 | 0.192 | 0.00155 |  |  |
| Weights SG-TD3 replay40k | 20260607_164103 | 15.279 | 8.888 | 13.557 | 8.142 | 0 | 0.152 | 0.00134 | 0.619 | 0.346 |
| Residual SG-TD3 param-noise replay40k | 20260607_164522 | 29.801 | 23.410 | 32.905 | -16.516 | 1 | 0.073 | 0.00104 | 0.449 | 0.037 |
| Markov SG-TD3 margin0.5 replay40k | 20260607_174217 | 29.931 | 23.540 | 31.820 | -13.022 | 5 | 0.066 | 0.00089 | 0.404 | 0.069 |
| Horizon SG-DQN replay40k | 20260607_155359 | 12.032 | 5.641 | 9.897 | 6.242 | 0 | 0.183 | 0.00095 | 0.135 | 0.054 |
| Dueling SG-DQN replay40k | 20260607_160052 | 11.368 | 4.977 | 11.104 | 5.836 | 0 | 0.190 | 0.00107 | 0.138 | 0.086 |

### Change Relative To Closest Mismatch Reference

| Family | Reference | Tail reward change | T85 MAE change | x24 MAE change | Neg post-warm change | Tail policy frac change | First-live policy frac change |
|---|---|---:|---:|---:|---:|---:|---:|
| weights | Weights SG-TD3 June 6 | -3.767 | -0.004 | 0.00039 | 0 | -0.095 | 0.053 |
| residual | Residual SG-TD3 June 6 no-param-noise | -1.370 | 0.003 | -0.00001 | -4 | -0.071 | -0.274 |
| markov | Markov SG-TD3 June 6 margin0 | 2.583 | 0.004 | -0.00027 | 3 | 0.038 | 0.032 |
| horizon | Horizon SG-DQN June 6 | -0.440 | 0.001 | -0.00003 | 0 | -0.084 | -0.002 |
| dueling horizon | Dueling SG-DQN June 4 wide | 5.014 | -0.024 | -0.00038 | -43 | -0.041 | -0.011 |

Positive reward change is good; negative tracking-MAE change is good.

### Configuration Audit

| Run | State mode | Action freeze | Actor freeze | Margin | Replay cap | Replay size | Exploration note |
|---|---|---:|---:|---:|---:|---:|---|
| Weights SG-TD3 replay40k | mismatch | 3 | 3 | 0.00 | 40000 | 40000 | gaussian 0.15 to 0.03; trace 74799 |
| Residual SG-TD3 param-noise replay40k | mismatch | 3 | 3 | 0.50 | 40000 | 40000 | param-noise 0.10 to 0.02; trace 74799 |
| Markov SG-TD3 margin0.5 replay40k | mismatch with Markov features | 3 | 3 | 0.50 | 40000 default | not saved | param-noise 0.10 to 0.02, trace not saved |
| Horizon SG-DQN replay40k | mismatch | 3 | n/a | 0.00 | 40000 | 40000 | epsilon |
| Dueling SG-DQN replay40k | mismatch | 3 | n/a | 0.00 | 40000 | 40000 | epsilon |

### Family Interpretation

**Weights SG-TD3.** Tail reward changed from 19.046 to 15.279. This is still the most defensible continuous supervisor if the goal is stable improvement: it does not directly add input residuals, and the current run keeps a low first-live policy fraction (0.346).

**Residual SG-TD3.** The residual policy is the highest-leverage actor because accepted actions directly perturb the MPC input. In this setup its tail reward is 29.801, with worst post-warm reward -16.516. Relative to June 6, the new settings reduced the number of negative post-warm episodes, but they did not remove the handoff shock. The margin and parameter noise should be judged by whether they improve the first 20 live episodes, not only the final tail, because a residual run can recover later after an early disturbance of the plant.

**Markov SG-TD3.** Markov's tail reward is 29.931. Its first-live policy fraction is 0.069, but negative post-warm episodes increased to 5. The worst episode has low policy acceptance, so at least part of the weakness looks like conservative under-correction or a supervisor/candidate scoring mismatch around the disturbed column state, not simply too much actor authority.

**Horizon SG-DQN.** The single-Q limited-grid DQN changed tail reward from 12.473 to 12.032. Because the action is a horizon recipe, the practical question is whether the learned policy concentrates around a small subset of stable recipes or continues to diffuse across the 39-action grid.

**Dueling SG-DQN.** The dueling network now has a like-for-like limited-grid mismatch result. Against the older wide-grid reference, tail reward is 11.368 versus 6.354. If this run still underperforms, the issue is probably not only the wide action set; the reward/gate signal for horizon selection is still weak.

### Worst Post-Warm Episodes

| Run | Worst subepisode | Reward | T85 MAE | x24 MAE | Policy frac | Median gate advantage |
|---|---:|---:|---:|---:|---:|---:|
| Weights SG-TD3 replay40k | 14 | 8.142 | 0.178 | 0.00146 | 0.545 | 1.159 |
| Residual SG-TD3 param-noise replay40k | 14 | -16.516 | 0.296 | 0.00297 | 0.098 | -66.807 |
| Markov SG-TD3 margin0.5 replay40k | 19 | -13.022 | 0.285 | 0.00292 | 0.128 | -109.227 |
| Horizon SG-DQN replay40k | 15 | 6.242 | 0.180 | 0.00109 | 0.058 | 0.000 |
| Dueling SG-DQN replay40k | 50 | 5.836 | 0.210 | 0.00119 | 0.075 | 0.000 |

### Horizon Candidate Diagnostics

| Run | Tail top pairs |
|---|---|
| Horizon SG-DQN replay40k | (6, 3) at 46.2 percent; (9, 5) at 3.2 percent; (6, 4) at 2.8 percent; (11, 3) at 2.5 percent; (6, 6) at 2.4 percent |
| Horizon SG-DQN June 6 | (6, 4) at 14.4 percent; (6, 6) at 13.1 percent; (6, 3) at 12.5 percent; (8, 6) at 7.0 percent; (6, 5) at 4.2 percent |
| Dueling SG-DQN replay40k | (6, 3) at 44.9 percent; (8, 7) at 25.9 percent; (10, 3) at 8.0 percent; (10, 5) at 2.2 percent; (11, 3) at 2.1 percent |
| Dueling SG-DQN June 4 wide | (6, 3) at 28.3 percent; (11, 11) at 19.0 percent; (12, 9) at 7.8 percent; (4, 2) at 5.8 percent; (9, 6) at 5.5 percent |

### Figures

- [fig_june7_tail_reward_negative_episodes.png](report/figures/distillation_sg_replay40k_warm3_20260607/fig_june7_tail_reward_negative_episodes.png)
- [fig_june7_current_vs_reference_delta.png](report/figures/distillation_sg_replay40k_warm3_20260607/fig_june7_current_vs_reference_delta.png)
- [fig_june7_episode_rewards.png](report/figures/distillation_sg_replay40k_warm3_20260607/fig_june7_episode_rewards.png)
- [fig_june7_gate_policy_fraction.png](report/figures/distillation_sg_replay40k_warm3_20260607/fig_june7_gate_policy_fraction.png)
- [fig_june7_horizon_tail_heatmaps.png](report/figures/distillation_sg_replay40k_warm3_20260607/fig_june7_horizon_tail_heatmaps.png)

### Next Experiment

Use weights SG-TD3 as the defensible continuous baseline unless residual clearly beats it without creating early negative episodes. For the next residual run, keep only one major change at a time: either keep the new parameter-noise setting and compare margins `0.0`, `0.25`, and `0.5`, or keep margin 0.5 and compare Gaussian versus parameter noise. The current bundle cannot fully separate replay-size effects from exploration and margin effects because they changed together.

For Markov, first inspect whether bad episodes have low policy acceptance. If yes, tune the Markov supervisor/gate scoring rather than making the actor more conservative; if no, reduce the margin/noise pressure. For horizons, keep `Np = 6..11`, `Nc = 3..Np` for one more paired DQN/dueling run, then narrow only if the tail heatmaps consistently concentrate below `Nc = 7`.

### Provenance Added For This Section

Files inspected:

- `report/distillation_sg_td3_sg_dqn_limited_horizon_results_2026_06_06.md`
- `report/scripts/analyze_distillation_sg_limited_horizons_20260606.py`
- `report/scripts/analyze_distillation_dueling_horizon_history_20260604.py`
- `distillation_RL_assisted_MPC_weights_supervisor_gated_td3_critic_warm_unified.py`
- `distillation_RL_assisted_MPC_residual_supervisor_gated_td3_critic_warm_unified.py`
- `distillation_RL_assisted_MPC_markov_supervisor_gated_td3_critic_warm_unified.py`
- `distillation_RL_assisted_MPC_horizons_supervisor_gated_dqn_unified.py`
- `distillation_RL_assisted_MPC_horizons_supervisor_gated_dueling_dqn_unified.py`
- `systems/distillation/notebook_params.py`
- `Distillation/Data/mpc_results_disturb_fluctuation.pickle`
- `Distillation/Results/distillation_weights_sg_td3_critic_warm3_margin0_sup001_gauss015_003_manual_off_disturb_fluctuation_mismatch/20260607_164103/input_data.pkl`
- `Distillation/Results/distillation_weights_sg_td3_critic_warm3_margin0_sup001_gauss015_003_manual_off_disturb_fluctuation_mismatch/20260606_075201/input_data.pkl`
- `Distillation/Results/distillation_residual_sg_td3_critic_warm3_margin05_paramnoise_manual_off_disturb_fluctuation_mismatch_no_rho/20260607_164522/input_data.pkl`
- `Distillation/Results/distillation_residual_sg_td3_critic_warm3_manual_off_disturb_fluctuation_mismatch_no_rho/20260606_074932/input_data.pkl`
- `Distillation/Results/distillation_markov_sg_td3_critic_warm3_margin05_softparamnoise_ls_else_mpc_shadow_disturb_fluctuation_mismatch/20260607_174217/input_data.pkl`
- `Distillation/Results/distillation_markov_sg_td3_critic_warm3_ls_else_mpc_shadow_disturb_fluctuation_mismatch_paramnoise/20260606_105404/input_data.pkl`
- `Distillation/Results/distillation_horizon_sg_dqn_critic_warm3_default_ofmpc_eps02_002_disturb_fluctuation_mismatch_np6_11_nc3_11/20260607_155359/input_data.pkl`
- `Distillation/Results/distillation_horizon_sg_dqn_critic_warm3_default_ofmpc_eps02_002_disturb_fluctuation_mismatch_np6_11_nc3_11/20260606_071451/input_data.pkl`
- `Distillation/Results/distillation_dueling_horizon_sg_dqn_critic_warm3_default_ofmpc_eps02_002_disturb_fluctuation_mismatch_np6_11_nc3_11/20260607_160052/input_data.pkl`
- `Distillation/Results/distillation_dueling_horizon_sg_dqn_critic_warm3_default_ofmpc_eps02_002_disturb_fluctuation_mismatch/20260604_105140/input_data.pkl`

Generated outputs:

- `report/figures/distillation_sg_replay40k_warm3_20260607/distillation_sg_replay40k_warm3_summary.csv`
- `report/figures/distillation_sg_replay40k_warm3_20260607/distillation_sg_replay40k_warm3_summary.json`
- `report/figures/distillation_sg_replay40k_warm3_20260607/distillation_sg_replay40k_warm3_episode_diagnostics.csv`
- `report/figures/distillation_sg_replay40k_warm3_20260607/distillation_sg_replay40k_warm3_horizon_pairs.csv`
- `report/figures/distillation_sg_replay40k_warm3_20260607/distillation_sg_replay40k_warm3_section.md`
- `report/figures/distillation_sg_replay40k_warm3_20260607/fig_june7_tail_reward_negative_episodes.png`
- `report/figures/distillation_sg_replay40k_warm3_20260607/fig_june7_current_vs_reference_delta.png`
- `report/figures/distillation_sg_replay40k_warm3_20260607/fig_june7_episode_rewards.png`
- `report/figures/distillation_sg_replay40k_warm3_20260607/fig_june7_gate_policy_fraction.png`
- `report/figures/distillation_sg_replay40k_warm3_20260607/fig_june7_horizon_tail_heatmaps.png`
