# Distillation Markov SG-TD3 Outcome Analysis

Date: 2026-06-03  
Result date: 2026-06-02  
Case study: Aspen C2 splitter distillation column  
Scenario: disturbed run with `disturbance_profile = "fluctuation"`

## Objective

This report analyzes the latest distillation Markov SG-TD3 run:

`Distillation/Results/distillation_markov_sg_td3_critic_warm3_ls_else_mpc_shadow_disturb_fluctuation_unified/20260602_192543/input_data.pkl`

The main question is whether supervisor-gated TD3 keeps the strong late Markov-correction performance seen in the June 1 TD3-full no-safeguard run while removing the severe early release crash.

Short answer:

Yes. The latest distillation Markov SG-TD3 result is much better than OF-MPC and much safer than TD3-full no-safeguard. SG-TD3 reaches tail-20 reward `26.62` versus OF-MPC `6.39`, and it avoids the TD3-full post-warm collapse. SG-TD3's worst first-20 post-warm reward is `3.67`, while TD3-full no-safeguard reached `-113.80`.

SG-TD3 does not fully match TD3-full in the final episode reward, but that is the expected safety-performance compromise. It preserves most of the tail improvement while keeping the supervisor active when the critic score does not support the actor.

## Files Inspected

Implementation and configuration:

- `distillation_RL_assisted_MPC_markov_supervisor_gated_td3_critic_warm_unified.py`
- `distillation_RL_assisted_MPC_markov_unified.py`
- `utils/markov_runner.py`
- `TD3Agent/supervisor_gated_agent.py`
- `utils/supervisor_gated_action.py`
- `systems/distillation/notebook_params.py`

Reports and change history:

- `report/distillation_polymer_markov_sg_td3_outcome_2026_06_02.md`
- `report/markov_dynamic_matrix_sg_td3_detailed_method_2026_06_02.md`
- `report/distillation_markov_safety_layer_audit_2026_05_29.md`
- `change-reports/2026-06-01_supervisor_gated_td3_addition.md`
- `change-reports/2026-06-01_distillation_sg_td3_critic_warm3_manual_off.md`

Result bundles:

| Role | Bundle |
|---|---|
| SG-TD3 run | `Distillation/Results/distillation_markov_sg_td3_critic_warm3_ls_else_mpc_shadow_disturb_fluctuation_unified/20260602_192543/input_data.pkl` |
| SG-TD3 compare | `Distillation/Results/distillation_compare_markov_sg_td3_critic_warm3_ls_else_mpc_shadow_disturb_fluctuation/20260602_192558/input_data.pkl` |
| TD3-full no-safeguard run | `Distillation/Results/distillation_markov_td3_disturb_fluctuation_td3_only_no_safeguard_current_reward_unified/20260601_214256/input_data.pkl` |
| TD3-full compare | `Distillation/Results/distillation_compare_markov_td3_disturb_fluctuation_td3_only_no_safeguard_current_reward/20260601_214316/input_data.pkl` |
| OF-MPC baseline | `Distillation/Data/mpc_results_disturb_fluctuation.pickle` |

Generated analysis artifacts:

| Artifact | Purpose |
|---|---|
| `report/scripts/analyze_distillation_markov_sg_td3_20260603.py` | Reproducible saved-bundle analysis |
| `report/figures/distillation_markov_sg_td3_20260603/reward_summary.csv` | Reward summary for SG-TD3, TD3-full, and OF-MPC |
| `report/figures/distillation_markov_sg_td3_20260603/tracking_summary.csv` | Physical tracking metrics |
| `report/figures/distillation_markov_sg_td3_20260603/tail_blockwise_tracking_summary.csv` | Tail metrics by setpoint block |
| `report/figures/distillation_markov_sg_td3_20260603/action_source_summary.csv` | SG-TD3 action-source fractions |
| `report/figures/distillation_markov_sg_td3_20260603/sg_gate_summary.csv` | Critic-gate score diagnostics |
| `report/figures/distillation_markov_sg_td3_20260603/episode_diagnostics.csv` | Per-episode reward, source, z, and score summaries |
| `report/figures/distillation_markov_sg_td3_20260603/manifest.json` | Machine-readable source and artifact manifest |

## What The Current Method Is Doing

The controller keeps the offset-free MPC backbone. The augmented linear prediction model is

$$ x_{\mathrm{aug},k+1}=A_{\mathrm{aug}}x_{\mathrm{aug},k}+B_{\mathrm{aug}}\Delta u_k,\qquad y_k=C_{\mathrm{aug}}x_{\mathrm{aug},k}. $$

The Markov correction changes the dynamic response matrix used by MPC. Let `G0` be the nominal lifted Markov matrix. The learned correction is parameterized by four coefficients:

$$ G(z_k)=G_0+\sum_{i=1}^{4}z_{i,k}G_i. $$

The TD3 actor outputs a normalized action:

$$ a_{\theta,k}\in[-1,1]^4,\qquad z_{\theta,k}=z_{\max}a_{\theta,k}. $$

For this run:

| Setting | Value |
|---|---|
| Agent | `sg_td3` |
| `z_bound` | `0.05` |
| Warm start | `10` subepisodes |
| Critic/action freeze | `3` post-warm subepisodes |
| Supervisor | `ls_else_mpc` |
| Live Markov hard safety | disabled |
| Shadow Markov safety | enabled |
| TD3 priority fallback | disabled |
| Behavioral cloning and handoff | disabled |
| Replay action | executed action with supervisor metadata |

The supervisor action is dynamic. If the LS Markov correction is available and accepted by the supervisor path, SG-TD3 compares the actor against LS. Otherwise it compares the actor against the nominal MPC-style supervisor.

The SG-TD3 score is a conservative twin-critic score with penalties for uncertainty, distance from supervisor, and movement from the previous action:

$$ S(s,a)=\min(Q_1(s,a),Q_2(s,a))-\rho_Q|Q_1(s,a)-Q_2(s,a)|-\kappa_{\mathrm{sup}}\|a-a_{\mathrm{sup}}\|_2^2-\kappa_{\mathrm{prev}}\|a-a_{\mathrm{prev}}\|_2^2. $$

The actor is executed only when

$$ S(s_k,a_{\theta,k})>S(s_k,a_{\mathrm{sup},k})+\epsilon_A. $$

For this run, `rho_Q = 0.5`, `kappa_sup = 0.02`, `kappa_prev = 0.01`, and `epsilon_A = 0`.

## Reward Results

![Reward curves](figures/distillation_markov_sg_td3_20260603/fig_reward_curves_sg_vs_td3full_vs_ofmpc.png)

![Reward summary](figures/distillation_markov_sg_td3_20260603/fig_reward_summary_bars.png)

| Method | Mean reward | Worst post-warm | Worst first 20 post-warm | Tail-20 reward | Final reward |
|---|---:|---:|---:|---:|---:|
| SG-TD3 | `18.799` | `-7.284` | `3.667` | `26.622` | `28.426` |
| TD3-full no safeguard | `-8.760` | `-113.802` | `-113.802` | `25.071` | `33.919` |
| OF-MPC | `7.719` | `4.488` | `9.070` | `6.391` | `6.926` |

The reward result has a clear mechanism:

- SG-TD3 improves strongly over OF-MPC in the tail.
- SG-TD3 avoids the catastrophic TD3-full release crash.
- TD3-full still has the best final episode reward, but only after a long unsafe recovery from early forced actor execution.

The key evidence is the first 20 post-warm window. TD3-full is forced through after warm start and collapses to `-113.802`. SG-TD3 mostly rejects the actor in the same release-sensitive period and its worst first-20 post-warm reward is `3.667`.

## Physical Tracking Results

Errors are computed in saved physical output coordinates against the active setpoint. `x24 ethane` is tray-24 ethane composition and `T85` is tray-85 temperature in the saved plant coordinate.

![Early release tracking](figures/distillation_markov_sg_td3_20260603/fig_early_release_tracking_zoom.png)

![Tail tracking](figures/distillation_markov_sg_td3_20260603/fig_tail_tracking_overlay.png)

| Method | Window | x24 MAE | T85 MAE | x24 RMSE | T85 RMSE | Mean abs scaled input move |
|---|---|---:|---:|---:|---:|---:|
| SG-TD3 | First 20 post-warm | `0.001378` | `0.1673` | `0.002809` | `0.3918` | `0.003906` |
| TD3-full | First 20 post-warm | `0.004444` | `0.6867` | `0.005764` | `0.9902` | `0.01917` |
| OF-MPC | First 20 post-warm | `0.001387` | `0.1707` | `0.002742` | `0.4078` | `0.003710` |
| SG-TD3 | Tail 20 | `0.001117` | `0.06691` | `0.002877` | `0.1684` | `0.004408` |
| TD3-full | Tail 20 | `0.001094` | `0.05258` | `0.002796` | `0.1519` | `0.006470` |
| OF-MPC | Tail 20 | `0.001545` | `0.1921` | `0.003078` | `0.4721` | `0.004242` |

The early-release window is the most important safety evidence. SG-TD3 tracks almost like OF-MPC immediately after warm start, while TD3-full is much worse on both outputs and uses much larger input movement.

The tail shows the safety-performance tradeoff. TD3-full is slightly better than SG-TD3 in tail tracking, but SG-TD3 is still far better than OF-MPC on both outputs. SG-TD3 reduces tail T85 MAE from `0.1921` to `0.0669` and reduces composition MAE from `0.001545` to `0.001117`.

### Blockwise Tail Tracking

![Tail blockwise MAE](figures/distillation_markov_sg_td3_20260603/fig_tail_blockwise_mae.png)

| Method | Block | x24 MAE | T85 MAE |
|---|---|---:|---:|
| SG-TD3 | SP1 | `0.000817` | `0.0510` |
| SG-TD3 | SP2 | `0.001417` | `0.0828` |
| TD3-full | SP1 | `0.000951` | `0.0395` |
| TD3-full | SP2 | `0.001237` | `0.0656` |
| OF-MPC | SP1 | `0.001508` | `0.1717` |
| OF-MPC | SP2 | `0.001582` | `0.2125` |

Both Markov-assisted methods improve both setpoint blocks relative to OF-MPC. SG-TD3 is more conservative than TD3-full in T85, but it still gives a large improvement in both SP1 and SP2.

## Action Source And Critic-Gate Diagnostics

![Action source fractions](figures/distillation_markov_sg_td3_20260603/fig_action_source_fractions.png)

| Window | TD3 accepted | SG LS supervisor | SG MPC supervisor | All supervisor |
|---|---:|---:|---:|---:|
| First 20 post-warm | `0.0204` | `0.0655` | `0.9141` | `0.9796` |
| Post-warm full | `0.1690` | `0.0767` | `0.7543` | `0.8310` |
| Tail 20 | `0.3808` | `0.1041` | `0.5151` | `0.6193` |

This is the behavior we wanted from distillation SG-TD3:

- During the fragile release period, SG-TD3 almost always executes the supervisor.
- In the tail, the TD3 actor receives more authority, but the supervisor is still active on `61.9%` of steps.
- The gate does not collapse into pure OF-MPC, and it does not force the actor through.

![SG advantage and scores](figures/distillation_markov_sg_td3_20260603/fig_sg_advantage_and_scores.png)

| Window | Advantage mean | Advantage q10 | Advantage q90 | Policy score greater than supervisor |
|---|---:|---:|---:|---:|
| First 20 post-warm | `-258.776` | `-552.035` | `-41.690` | `0.0204` |
| Post-warm full | `-125.794` | `-445.022` | `9.646` | `0.1690` |
| Tail 20 | `3.527` | `-17.356` | `26.884` | `0.3808` |

The critic-gate scores explain the action-source fractions. In the first 20 post-warm episodes, the actor score is much lower than the supervisor score, so the actor is almost never selected. In the tail, the score difference becomes slightly positive on average and the actor selection fraction rises.

## Markov Correction And Input Movement

![Markov correction norm and input move](figures/distillation_markov_sg_td3_20260603/fig_markov_correction_norm_and_input_move.png)

The tracking table already shows the key input-movement result:

- First 20 post-warm mean abs scaled input move is `0.00391` for SG-TD3, close to OF-MPC's `0.00371`.
- TD3-full uses `0.01917` in the same window, which is almost five times larger.
- In the tail, SG-TD3 uses `0.00441`, close to OF-MPC's `0.00424`, while TD3-full uses `0.00647`.

This supports the interpretation that SG-TD3 improves Markov adaptation without relying on the large early input movements that caused the TD3-full release crash.

## Main Interpretation

### SG-TD3 Solves The Release Problem

The June 1 TD3-full no-safeguard run proved that the learned Markov actor has high late upside, but it also proved that forced early actor execution is unsafe. The June 2 SG-TD3 run keeps the useful part and removes the worst failure mode.

The most important number is not only tail reward. It is the early-release contrast:

- TD3-full worst first-20 post-warm reward: `-113.802`
- SG-TD3 worst first-20 post-warm reward: `3.667`
- OF-MPC worst first-20 post-warm reward: `9.070`

SG-TD3 is slightly worse than OF-MPC in the worst release episode, but it is in the same regime. TD3-full is not.

### SG-TD3 Preserves Most Of The Late Markov Benefit

SG-TD3 tail-20 reward is `26.622`, slightly higher than the TD3-full tail-20 reward of `25.071`. TD3-full has a stronger final episode, `33.919` versus SG-TD3's `28.426`, but SG-TD3 still beats OF-MPC by a wide margin.

The tail tracking confirms that this is not just reward shaping:

- T85 tail MAE improves from `0.1921` to `0.0669`.
- Composition tail MAE improves from `0.001545` to `0.001117`.
- Tail input movement remains close to OF-MPC.

### The Supervisor Is Still Doing Real Work

The tail actor fraction is only `38.1%`. That is a strength, not a weakness, for this run. The gate is selecting TD3 when the critic score supports it and keeping the LS-or-MPC supervisor otherwise.

This matters because Markov corrections change the response model used inside MPC. They are more structurally risky than a small residual input correction. A useful distillation Markov controller needs selective authority, not blanket actor execution.

## Bugs, Inconsistencies, Or Risks Found

1. The run is single-seed evidence.
   The result is strong, but it should not be treated as final publication-level ranking until repeated with at least several seeds or repeated run timestamps.

2. The old hard Markov safety layers are shadow-only.
   This report evaluates the live SG critic gate, not live z-safety or TD3-priority fallback. Shadow diagnostics remain useful, but they are not live constraints.

3. The SG-TD3 run is not pure TD3.
   The controller is a hybrid controller. Its performance should be described as supervisor-gated Markov control, not as an unrestricted learned Markov actor.

4. The final TD3-full episode remains stronger than SG-TD3.
   SG-TD3 is better as a safety-performance compromise. It is not simply better on every metric.

5. Distillation T85 labels should be interpreted in the saved plant coordinate.
   The saved values are negative, so this report avoids making absolute thermodynamic-unit claims beyond the stored output coordinate.

## Literature Connections

No new external citations were added. The analysis is grounded in local saved results and previous repo reports. Conceptually, the result supports the safe supervised-RL pattern used elsewhere in this project: compare a learned actor against a stabilizing supervisor and execute the actor only when a conservative value score supports it.

This report should be connected later to literature on safe RL for process control, MPC-RL integration, and critic-based policy selection only after the citations are verified and added to the manuscript bibliography.

## Recommended Next Experiment

The next experiment should be a reproducibility and authority study, not another broad method change.

1. Repeat distillation Markov SG-TD3 with the same configuration for at least three seeds or timestamps.
2. Keep the current `z_bound = 0.05`, current reward, `ls_else_mpc` supervisor, and critic-warm-3 release.
3. Save the same action-source, SG score, z norm, input move, and tracking diagnostics.
4. Add a focused sensitivity run with a slightly more permissive gate only after the repeated run confirms release safety.

Recommended success criteria:

| Metric | Target |
|---|---:|
| Worst first 20 post-warm reward | above `0`, or at least no lower than OF-MPC by more than `10` reward units |
| Tail-20 reward | above `20` |
| Final reward | above `20` |
| Tail TD3 accepted fraction | between `0.25` and `0.60` |
| First 20 post-warm TD3 accepted fraction | below `0.10` |
| Tail T85 MAE | below `0.10` |
| Tail x24 MAE | below `0.0015` |

The most useful ablation after repeats is a gate sensitivity study:

- current gate: `rho_Q = 0.5`, `kappa_sup = 0.02`, `kappa_prev = 0.01`, `epsilon_A = 0`
- slightly more actor authority: reduce `kappa_sup` or `kappa_prev` by half
- more conservative release: require a small positive `epsilon_A`

The gate should not be tuned only for final reward. It should be tuned against early-release reward, tail actor fraction, T85 MAE, and input movement.

## Remaining Uncertainty

The main uncertainty is repeatability. The current run is strong enough to change the interpretation of the Markov family: SG-TD3 is now the best-looking distillation Markov safety-performance compromise in the saved results. But it remains a single run.

Another uncertainty is whether the critic is genuinely learning a reliable action selector or whether this run benefited from a favorable training trajectory. The score diagnostics are encouraging because the policy score is very negative relative to the supervisor early and mildly positive in the tail. Repeated runs should verify that this score transition is stable.

## Files Changed

- Created `report/scripts/analyze_distillation_markov_sg_td3_20260603.py`
- Created `report/distillation_markov_sg_td3_outcome_2026_06_03.md`
- Generated local ignored artifacts under `report/figures/distillation_markov_sg_td3_20260603/`

## How To Verify

Run:

```powershell
C:\Users\hamediaa\.conda\envs\rl-env\python.exe report\scripts\analyze_distillation_markov_sg_td3_20260603.py
```

Then inspect:

- `report/figures/distillation_markov_sg_td3_20260603/manifest.json`
- `report/figures/distillation_markov_sg_td3_20260603/reward_summary.csv`
- `report/figures/distillation_markov_sg_td3_20260603/tracking_summary.csv`
- `report/figures/distillation_markov_sg_td3_20260603/action_source_summary.csv`
