# Distillation Column Latest Runner Analysis

Generated on 2026-06-09 from saved result bundles only. Aspen was not relaunched.

## Scope

This report analyzes the latest saved disturbance/fluctuation mismatch batch for the active distillation column runners: weights SG-TD3, residual SG-TD3, Markov SG-TD3, horizon SG-DQN, and dueling horizon SG-DQN. The disturbed OF-MPC trajectory in `Distillation/Data/mpc_results_disturb_fluctuation.pickle` is used as the baseline. Older standard, nominal, matrix, structured-matrix, residual-TD7, and combined artifacts remain in the repository, but they are not mixed into the headline comparison because they were produced under different runner families or older configurations.

## Executive Result

The best tail reward in the latest active batch is **Residual SG-TD3 param-noise replay40k** with tail reward 30.468, which is 24.077 above OF-MPC. The best no-negative-post-warm run is **Residual SG-TD3 param-noise replay40k** with tail reward 30.468.

The main scientific tradeoff is still authority versus reliability, but the latest residual run is the strongest current result: it has the best tail reward and avoids negative post-warm episodes. Markov remains almost as strong in the tail but has the visible negative post-warm episode. Weights is the lower-authority continuous reference because its fallback is the identity multiplier. The horizon agents are stable in this batch but still weaker than the continuous SG-TD3 families.

| Run | Timestamp | Tail reward | Delta vs OF-MPC | Final reward | Worst post-warm | Neg post-warm | T85 MAE | x24 MAE | Tail policy frac | First-live policy frac |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OF-MPC | Data | 6.391 | 0.000 | 6.926 | 4.488 | 0 | 0.192 | 0.00155 |  |  |
| Weights SG-TD3 replay40k | 20260608_181133 | 17.128 | 10.737 | 17.140 | 6.354 | 0 | 0.146 | 0.00122 | 0.620 | 0.280 |
| Residual SG-TD3 param-noise replay40k | 20260608_183743 | 30.468 | 24.077 | 32.754 | 5.703 | 0 | 0.074 | 0.00097 | 0.478 | 0.013 |
| Markov SG-TD3 margin0.5 replay40k | 20260608_194745 | 30.209 | 23.818 | 33.661 | -10.194 | 1 | 0.051 | 0.00113 | 0.399 | 0.040 |
| Horizon SG-DQN replay40k | 20260608_180237 | 11.803 | 5.412 | 11.919 | 7.071 | 0 | 0.188 | 0.00099 | 0.230 | 0.059 |
| Dueling SG-DQN replay40k | 20260608_180736 | 11.368 | 4.977 | 11.104 | 5.836 | 0 | 0.190 | 0.00107 | 0.138 | 0.086 |

![Tail reward and negative post-warm episodes](figures/distillation_all_runners_latest_20260609/fig_latest_tail_reward_negative_episodes.png)

## Method Summary

The distillation plant outputs are tray-24 ethane composition and tray-85 temperature, and the manipulated inputs are reflux flow and reboiler duty. The baseline controller is an offset-free linear MPC built from the column identification artifacts. All active RL runners keep MPC as the move generator or fallback rather than replacing it with direct black-box control.

$$ \min_{\Delta U} \sum_{i=1}^{N_p} (y_{k+i}-y_{\mathrm{sp},k+i})^\top Q (y_{k+i}-y_{\mathrm{sp},k+i}) + \sum_{i=0}^{N_c-1} \Delta u_{k+i}^\top R \Delta u_{k+i}. $$

The comparison rescored every saved trajectory with the current distillation reward helper. In compact form, the reward is a tracking-and-move penalty plus a smooth in-band bonus:

$$ r_t = -e_t^\top Q e_t - \Delta u_t^\top R \Delta u_t - \ell_{\mathrm{band}}(e_t,y_{\mathrm{sp},t}) + b_{\mathrm{inside}}(e_t,y_{\mathrm{sp},t}). $$

The exact rescoring code computes scaled output errors, scaled input moves, a physical tolerance band using `k_rel = [0.3, 0.01]`, `band_floor_phys = [0.003, 0.2]`, `Q_diag = [3.7e4, 2.0e4]`, and `R_diag = [2.5e3, 2.5e3]`, then adds a geometric in-band bonus. This avoids comparing runs on incompatible logged reward revisions.

The SG-TD3 runners use a critic-based execution gate:

$$ a_{\mathrm{exec}} = a_{\mathrm{rl}}\ \mathrm{if}\ S(a_{\mathrm{rl}})-S(a_{\mathrm{sup}})>m,\ \mathrm{else}\ a_{\mathrm{sup}}. $$

For weights, `a_sup` is the identity multiplier. For residual, it is zero residual. For Markov, it is the LS-or-MPC Markov correction. The horizon agents use the same idea over the discrete triangular action set `Np = 6..11`, `Nc = 3..Np`, with `(6, 3)` as the OF-MPC supervisor recipe.

| Run | State mode | Action freeze | Actor freeze | Margin | Replay cap | Replay size | Exploration |
|---|---|---:|---:|---:|---:|---:|---|
| Weights SG-TD3 replay40k | mismatch | 3 | 3 | 0.00 | 40000 | 40000 | gaussian 0.15 to 0.03, trace 74799 |
| Residual SG-TD3 param-noise replay40k | mismatch | 3 | 3 | 0.50 | 40000 | 40000 | param-noise 0.10 to 0.02, trace 74799 |
| Markov SG-TD3 margin0.5 replay40k | mismatch with Markov features | 3 | 3 | 0.50 | 40000 default | not saved | param-noise 0.10 to 0.02, trace not saved |
| Horizon SG-DQN replay40k | mismatch | 3 | n/a | 0.00 | 40000 | 40000 | epsilon |
| Dueling SG-DQN replay40k | mismatch | 3 | n/a | 0.00 | 40000 | 40000 | epsilon |

## Changes From Closest References

Positive reward change is good. Negative T85 or x24 MAE change is good. The reference rows are the closest saved mismatch runs from the preceding analysis batches, not a full hyperparameter sweep.

| Family | Reference | Tail reward change | T85 MAE change | x24 MAE change | Neg post-warm change | Tail policy frac change | First-live policy frac change |
|---|---|---:|---:|---:|---:|---:|---:|
| weights | Weights SG-TD3 June 6 | -1.919 | -0.010 | 0.00026 | 0 | -0.093 | -0.014 |
| residual | Residual SG-TD3 June 6 no-param-noise | -0.703 | 0.004 | -0.00008 | -5 | -0.042 | -0.299 |
| markov | Markov SG-TD3 June 6 margin0 | 2.861 | -0.011 | -0.00003 | -1 | 0.033 | 0.003 |
| horizon | Horizon SG-DQN June 6 | -0.669 | 0.005 | 0.00001 | 0 | 0.012 | 0.003 |
| dueling horizon | Dueling SG-DQN June 4 wide | 5.014 | -0.024 | -0.00038 | -43 | -0.041 | -0.011 |

![Change relative to closest mismatch reference](figures/distillation_all_runners_latest_20260609/fig_latest_current_vs_reference_delta.png)

## Learning And Release Evidence

The episode traces separate warm start, critic-only handoff, and live policy release. The worst-post-warm table is used because a high final tail reward can hide a damaging early live episode.

![Episode reward histories](figures/distillation_all_runners_latest_20260609/fig_latest_episode_rewards.png)

![Supervisor-gate policy fraction](figures/distillation_all_runners_latest_20260609/fig_latest_gate_policy_fraction.png)

| Run | Worst subepisode | Reward | T85 MAE | x24 MAE | Policy frac | Median gate advantage |
|---|---:|---:|---:|---:|---:|---:|
| Weights SG-TD3 replay40k | 17 | 6.354 | 0.191 | 0.00165 | 0.285 | -2.809 |
| Residual SG-TD3 param-noise replay40k | 23 | 5.703 | 0.208 | 0.00217 | 0.035 | -295.133 |
| Markov SG-TD3 margin0.5 replay40k | 16 | -10.194 | 0.290 | 0.00256 | 0.107 | -99.613 |
| Horizon SG-DQN replay40k | 16 | 7.071 | 0.180 | 0.00110 | 0.052 | 0.000 |
| Dueling SG-DQN replay40k | 50 | 5.836 | 0.210 | 0.00119 | 0.075 | 0.000 |

## Horizon Diagnostics

Both horizon agents now have limited-grid mismatch runs. The heatmaps show whether the learned policy is concentrating on a small stable recipe subset or simply exploring the 39 valid recipes.

| Run | Tail top pairs |
|---|---|
| Horizon SG-DQN replay40k | (6, 6) at 8.2 percent; (6, 3) at 7.8 percent; (7, 4) at 5.1 percent; (11, 3) at 4.3 percent; (11, 8) at 4.2 percent |
| Horizon SG-DQN June 6 | (6, 4) at 14.4 percent; (6, 6) at 13.1 percent; (6, 3) at 12.5 percent; (8, 6) at 7.0 percent; (6, 5) at 4.2 percent |
| Dueling SG-DQN replay40k | (6, 3) at 44.9 percent; (8, 7) at 25.9 percent; (10, 3) at 8.0 percent; (10, 5) at 2.2 percent; (11, 3) at 2.1 percent |
| Dueling SG-DQN June 4 wide | (6, 3) at 28.3 percent; (11, 11) at 19.0 percent; (12, 9) at 7.8 percent; (4, 2) at 5.8 percent; (9, 6) at 5.5 percent |

![Horizon tail heatmaps](figures/distillation_all_runners_latest_20260609/fig_latest_horizon_tail_heatmaps.png)

## Interpretation

**Weights SG-TD3.** Tail reward is 17.128, with 0 negative post-warm episodes. This is the conservative continuous reference: it changes the MPC tradeoff rather than directly adding input corrections, but it does not match the latest residual tail reward.

**Residual SG-TD3.** Tail reward is 30.468, and the worst post-warm reward is 5.703. This is the best latest run, not only the highest-reward run. Because accepted residuals are plant-facing input corrections, it still needs seed replication before it should be treated as robust.

**Markov SG-TD3.** Tail reward is 30.209, with 1 negative post-warm episode. The mechanism is model correction rather than direct input correction, so weak episodes should be diagnosed through supervisor/candidate scoring and LS-or-MPC correction quality.

**Horizon SG-DQN.** Tail reward is 11.803. The action is a horizon recipe, so the method is low-authority and stable here, but still far behind the best continuous supervisors.

**Dueling SG-DQN.** Tail reward is 11.368. Dueling helped create a saved limited-grid mismatch run, but the performance gap to residual and Markov shows that value decomposition alone is not enough for this column scenario.

## Bugs, Inconsistencies, And Risks

- The June 7 analysis folder was stale after the June 8 reruns. This report uses the latest saved timestamps and writes a new dated output folder.
- Markov replay size is not saved in the same way as the other current bundles, so the report records it as `not saved` instead of inferring it.
- The comparison is single-seed per latest runner. Tail rankings should be treated as batch evidence, not a statistical conclusion.
- Some older result families are not included because their saved configs are not like-for-like with the latest SG warm-3 mismatch batch.

## Literature Connections

No local BibTeX file was found during this pass, so no new formal citations were added. The interpretation follows the local `StatsControl2026/rl_assisted_mpc_algorithm_slides_2026_06_02.tex` framing: RL changes an MPC design knob, while MPC and the supervisor gate define the executable controller envelope. The residual-risk interpretation also matches the local method notes that residual actions are direct input corrections, whereas weights and horizon changes act through MPC.

## Recommended Next Experiments

1. Treat residual SG-TD3 as the current candidate winner and weights SG-TD3 as the conservative continuous baseline. Run two more seeds for both with the same warm-3 and replay-40k settings, and use tail reward, T85 MAE, x24 MAE, and negative post-warm episodes as the acceptance metrics.
2. For residual SG-TD3, isolate margin from exploration. Keep parameter noise fixed and sweep `advantage_margin` over `0.0`, `0.25`, and `0.5`, or keep margin fixed and compare Gaussian versus parameter noise.
3. For Markov SG-TD3, inspect low-reward episodes by gate advantage and policy fraction. If bad episodes have low policy acceptance, tune the LS-or-MPC supervisor or scoring terms before making the actor more conservative.
4. For horizon and dueling horizon, keep the triangular limited grid for one more paired run. Narrow only if the tail heatmaps again concentrate around low control horizons.

## Provenance

Files inspected:

- `report/scripts/analyze_distillation_sg_replay40k_warm3_20260607.py`
- `report/scripts/analyze_distillation_sg_limited_horizons_20260606.py`
- `report/scripts/analyze_distillation_dueling_horizon_history_20260604.py`
- `distillation_RL_assisted_MPC_weights_supervisor_gated_td3_critic_warm_unified.py`
- `distillation_RL_assisted_MPC_residual_supervisor_gated_td3_critic_warm_unified.py`
- `distillation_RL_assisted_MPC_markov_supervisor_gated_td3_critic_warm_unified.py`
- `distillation_RL_assisted_MPC_horizons_supervisor_gated_dqn_unified.py`
- `distillation_RL_assisted_MPC_horizons_supervisor_gated_dueling_dqn_unified.py`
- `systems/distillation/config.py`
- `systems/distillation/notebook_params.py`
- `systems/distillation/labels.py`
- `StatsControl2026/rl_assisted_mpc_algorithm_slides_2026_06_02.tex`
- `Distillation/Data/mpc_results_disturb_fluctuation.pickle`
- `Distillation/Results/distillation_weights_sg_td3_critic_warm3_margin0_sup001_gauss015_003_manual_off_disturb_fluctuation_mismatch/20260608_181133/input_data.pkl`
- `Distillation/Results/distillation_weights_sg_td3_critic_warm3_margin0_sup001_gauss015_003_manual_off_disturb_fluctuation_mismatch/20260606_075201/input_data.pkl`
- `Distillation/Results/distillation_residual_sg_td3_critic_warm3_margin05_paramnoise_manual_off_disturb_fluctuation_mismatch_no_rho/20260608_183743/input_data.pkl`
- `Distillation/Results/distillation_residual_sg_td3_critic_warm3_manual_off_disturb_fluctuation_mismatch_no_rho/20260606_074932/input_data.pkl`
- `Distillation/Results/distillation_markov_sg_td3_critic_warm3_margin05_softparamnoise_ls_else_mpc_shadow_disturb_fluctuation_mismatch/20260608_194745/input_data.pkl`
- `Distillation/Results/distillation_markov_sg_td3_critic_warm3_ls_else_mpc_shadow_disturb_fluctuation_mismatch_paramnoise/20260606_105404/input_data.pkl`
- `Distillation/Results/distillation_horizon_sg_dqn_critic_warm3_default_ofmpc_eps02_002_disturb_fluctuation_mismatch_np6_11_nc3_11/20260608_180237/input_data.pkl`
- `Distillation/Results/distillation_horizon_sg_dqn_critic_warm3_default_ofmpc_eps02_002_disturb_fluctuation_mismatch_np6_11_nc3_11/20260606_071451/input_data.pkl`
- `Distillation/Results/distillation_dueling_horizon_sg_dqn_critic_warm3_default_ofmpc_eps02_002_disturb_fluctuation_mismatch_np6_11_nc3_11/20260608_180736/input_data.pkl`
- `Distillation/Results/distillation_dueling_horizon_sg_dqn_critic_warm3_default_ofmpc_eps02_002_disturb_fluctuation_mismatch/20260604_105140/input_data.pkl`

Generated outputs:

- `report/distillation_all_runners_latest_analysis_2026_06_09.md`
- `report/figures/distillation_all_runners_latest_20260609/distillation_all_runners_latest_summary.csv`
- `report/figures/distillation_all_runners_latest_20260609/distillation_all_runners_latest_summary.json`
- `report/figures/distillation_all_runners_latest_20260609/distillation_all_runners_latest_episode_diagnostics.csv`
- `report/figures/distillation_all_runners_latest_20260609/distillation_all_runners_latest_horizon_pairs.csv`
- `report/figures/distillation_all_runners_latest_20260609/fig_latest_tail_reward_negative_episodes.png`
- `report/figures/distillation_all_runners_latest_20260609/fig_latest_current_vs_reference_delta.png`
- `report/figures/distillation_all_runners_latest_20260609/fig_latest_episode_rewards.png`
- `report/figures/distillation_all_runners_latest_20260609/fig_latest_gate_policy_fraction.png`
- `report/figures/distillation_all_runners_latest_20260609/fig_latest_horizon_tail_heatmaps.png`
