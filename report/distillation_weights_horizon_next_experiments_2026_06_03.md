# Distillation Weights And Horizon Rerun Analysis

Date: 2026-06-03

Case study: Aspen Dynamics C2 splitter distillation column

Scenario: `run_mode = "disturb"`, `disturbance_profile = "fluctuation"`, `state_mode = "mismatch"`

## Objective

This report reviews the three fresh distillation reruns:

- SG-TD3 weights with Gaussian exploration.
- Standard DDQN horizon with epsilon-greedy exploration.
- Dueling DDQN horizon with epsilon-greedy exploration.

The goal is to decide whether the next experiment should be SG-DQN, a horizon-range change, another reward change, or further weight tuning.

## Executive Summary

The weights rerun is the cleanest improvement. Gaussian exploration plus the relaxed SG gate raises the SG-TD3 weights tail reward from `14.37` to `18.41`, and the final reward from `13.74` to `16.20`. It also keeps every post-warm episode nonnegative. The gate is no longer frozen near identity: tail TD3 policy selection rises from `32.3%` to `57.9%`.

The horizon reruns are better than the June 2 NoisyNet runs, but they are still not safe. Standard DDQN improves tail reward to `8.65`, but it has a severe post-warm crash at `-22.86` and `13` negative post-warm episodes. Dueling DDQN improves tail reward to `9.29` and final reward to `15.22`, but it has `30` negative post-warm episodes.

The intended epsilon schedule did not reach `0.02`. Both saved epsilon traces end near `0.133`. The likely reason is scale mismatch: the agent decays epsilon by live action-selection calls, while `eps_decay_steps = 50000` was chosen more like a plant-step count. With a decision interval of `4`, the run only reaches about `18.6k` DQN action calls, so the schedule is internally consistent but too slow.

My recommendation is:

1. Implement SG-DQN for the horizon family before changing the horizon range.
2. Use dueling SG-DQN as the primary next run, because the latest dueling run has the best horizon tail and final reward.
3. Run standard SG-DQN as the paired ablation if time allows, because standard DDQN has fewer negative episodes but one much deeper crash.
4. Correct epsilon decay to action-call scale, roughly `0.20 -> 0.02` over `18000` to `20000` DQN action calls for this `80k` plant-step setup.
5. Keep the 87-action grid for the first SG-DQN ablation, then prune only after the gate logs show which recipes survive.

## Files Inspected

Implementation files:

- `distillation_RL_assisted_MPC_weights_supervisor_gated_td3_critic_warm_unified.py`
- `distillation_RL_assisted_MPC_horizons_unified.py`
- `distillation_RL_assisted_MPC_horizons_dueling_unified.py`
- `systems/distillation/notebook_params.py`
- `utils/weights_runner.py`
- `utils/horizon_runner.py`
- `utils/horizon_runner_dueling.py`
- `DQN/dqn_agent.py`
- `DuelingDQN/dueling_dqn_agent.py`
- `TD3Agent/supervisor_gated_agent.py`
- `TD3Agent/supervisor_replay_buffer.py`

Reports and history:

- `report/distillation_weights_sg_td3_diagnosis_2026_06_02.md`
- `report/distillation_horizon_dqn_dueling_diagnosis_2026_06_02.md`
- `report/distillation_markov_sg_td3_outcome_2026_06_03.md`
- `change-reports/2026-06-02_distillation_horizon_epsilon_greedy.md`

Result bundles:

- `Distillation/Results/distillation_weights_sg_td3_critic_warm3_margin0_sup001_gauss015_003_manual_off_disturb_fluctuation_mismatch/20260603_124214/input_data.pkl`
- `Distillation/Results/distillation_compare_weights_sg_td3_critic_warm3_margin0_sup001_gauss015_003_manual_off_disturb_fluctuation/20260603_124224/input_data.pkl`
- `Distillation/Results/distillation_horizon_disturb_fluctuation_mismatch_unified/20260603_130635/input_data.pkl`
- `Distillation/Results/distillation_compare_horizon_disturb_fluctuation_mismatch/20260603_130644/input_data.pkl`
- `Distillation/Results/distillation_dueling_horizon_disturb_fluctuation_mismatch_unified/20260603_130106/input_data.pkl`
- `Distillation/Results/distillation_compare_dueling_horizon_disturb_fluctuation_mismatch/20260603_130117/input_data.pkl`
- `Distillation/Data/mpc_results_disturb_fluctuation.pickle`

Previous reference bundles:

- `Distillation/Results/distillation_weights_sg_td3_critic_warm3_manual_off_disturb_fluctuation_mismatch/20260602_140102/input_data.pkl`
- `Distillation/Results/distillation_horizon_disturb_fluctuation_mismatch_unified/20260602_144031/input_data.pkl`
- `Distillation/Results/distillation_dueling_horizon_disturb_fluctuation_mismatch_unified/20260602_144134/input_data.pkl`

## Analysis Artifacts

The reproducible local analysis command is:

```powershell
C:\Users\hamediaa\.conda\envs\rl-env\python.exe report\scripts\analyze_distillation_latest_weights_horizon_20260603.py
```

Generated local artifacts:

- `report/figures/distillation_latest_weights_horizon_20260603/reward_summary.csv`
- `report/figures/distillation_latest_weights_horizon_20260603/tracking_summary.csv`
- `report/figures/distillation_latest_weights_horizon_20260603/horizon_summary.csv`
- `report/figures/distillation_latest_weights_horizon_20260603/horizon_top_pairs_tail.csv`
- `report/figures/distillation_latest_weights_horizon_20260603/weights_sg_summary.csv`
- `report/figures/distillation_latest_weights_horizon_20260603/manifest.json`

![Reward summary](figures/distillation_latest_weights_horizon_20260603/fig_reward_summary_latest_vs_previous.png)

![Reward curves](figures/distillation_latest_weights_horizon_20260603/fig_reward_curves_latest_three_vs_previous.png)

![Tail tracking MAE](figures/distillation_latest_weights_horizon_20260603/fig_tail_tracking_mae_latest_vs_previous.png)

![Horizon usage](figures/distillation_latest_weights_horizon_20260603/fig_horizon_usage_and_stability.png)

![Weights SG gate](figures/distillation_latest_weights_horizon_20260603/fig_weights_sg_gate_gaussian_vs_previous.png)

## Current Method

The controlled output is:

$$ y_k = [x_{24,\mathrm{C2H6},k}, T_{85,k}]^\top. $$

The manipulated input is:

$$ u_k = [F_{\mathrm{reflux},k}, Q_{\mathrm{reb},k}]^\top. $$

All three methods keep the same offset-free MPC backbone. The baseline controller solves a constrained receding-horizon optimization in scaled deviation coordinates, then the Aspen plant advances in physical coordinates.

The weight agent changes the MPC penalty multipliers:

$$ m_k = [m_{Q1,k}, m_{Q2,k}, m_{R1,k}, m_{R2,k}]^\top. $$

The weighted MPC cost is:

$$ J_{m_k} = \sum_{j=1}^{N_p} e_{k+j}^{\top}\mathrm{diag}(m_{Q,k}\odot Q_0)e_{k+j} + \sum_{j=0}^{N_c-1}\Delta u_{k+j}^{\top}\mathrm{diag}(m_{R,k}\odot R_0)\Delta u_{k+j}. $$

SG-TD3 compares the actor proposal with the identity multiplier supervisor. The conservative score is:

$$ S(s,a) = \min(Q_1(s,a), Q_2(s,a)) - \rho_Q |Q_1(s,a)-Q_2(s,a)| - \kappa_{\mathrm{sup}}\|a-a_{\mathrm{sup}}\|_2^2 - \kappa_{\mathrm{prev}}\|a-a_{\mathrm{prev}}\|_2^2. $$

The latest weights wrapper uses Gaussian exploration:

$$ \sigma: 0.15 \rightarrow 0.03. $$

It also uses the relaxed gate:

$$ \epsilon_A = 0,\quad \rho_Q=0.5,\quad \kappa_{\mathrm{sup}}=0.01,\quad \kappa_{\mathrm{prev}}=0.01. $$

The horizon agents choose a discrete MPC recipe:

$$ a_k = (N_{p,k},N_{c,k}) \in \mathcal{A}_H. $$

The current grid has `87` feasible recipes:

$$ N_p \in \{4,\ldots,14\},\quad N_c \in \{2,\ldots,13\},\quad N_c \le N_p. $$

The default recipe is `(6, 3)`. The standard agent uses Double DQN:

$$ y_k^{\mathrm{DDQN}} = r_k + \gamma Q_{\bar{\theta}}(s_{k+1}, \arg\max_a Q_{\theta}(s_{k+1},a)). $$

The dueling agent uses:

$$ Q_{\theta}(s,a) = V_{\theta}(s) + A_{\theta}(s,a) - \frac{1}{|\mathcal{A}_H|}\sum_b A_{\theta}(s,b). $$

The latest horizon reruns use epsilon-greedy exploration instead of NoisyNet. The intended schedule was `0.20 -> 0.02`, but the saved traces end at `0.133`.

## Reward Results

All reward values below come from the compare bundles. This keeps OF-MPC and RL under the same current reward definition.

| Method | Tail-20 reward | Final reward | Worst post-warm | Worst first-20 post-warm | Tail gain vs OF-MPC | Tail gain vs previous |
|---|---:|---:|---:|---:|---:|---:|
| OF-MPC | `6.391` | `6.926` | `4.488` | `9.070` | NA | NA |
| SG-TD3 weights gaussian | `18.412` | `16.195` | `1.627` | `1.627` | `+12.021` | `+4.037` |
| DDQN horizon epsilon | `8.653` | `9.736` | `-22.864` | `-16.901` | `+2.262` | `+1.317` |
| Dueling horizon epsilon | `9.292` | `15.222` | `-9.422` | `1.996` | `+2.901` | `+5.299` |
| SG-TD3 weights previous | `14.375` | `13.739` | `3.456` | `8.347` | `+7.984` | NA |
| DDQN horizon previous | `7.336` | `8.915` | `-32.409` | `-1.338` | `+0.945` | NA |
| Dueling horizon previous | `3.993` | `1.561` | `-8.831` | `0.647` | `-2.398` | NA |

The important separation is:

- Weights SG-TD3 is now clearly above OF-MPC and above the previous weights SG-TD3 run.
- Horizon DDQN is improved in tail reward, but its early release is worse than before.
- Dueling horizon is much better in tail and final reward, but it still has mid-run unsafe episodes.

Negative post-warm episode counts:

| Method | Negative post-warm episodes | Worst episode after warm |
|---|---:|---:|
| OF-MPC | `0` | `4.488` |
| SG-TD3 weights gaussian | `0` | `1.627` |
| DDQN horizon epsilon | `13` | `-22.864` |
| Dueling horizon epsilon | `30` | `-9.422` |

This is the main reason SG-DQN remains the next missing experiment.

## Physical Tracking

Tail tracking MAE:

| Method | x24 ethane MAE | T85 MAE |
|---|---:|---:|
| OF-MPC | `0.001545` | `0.1921` |
| SG-TD3 weights gaussian | `0.001216` | `0.1545` |
| DDQN horizon epsilon | `0.001052` | `0.2063` |
| Dueling horizon epsilon | `0.001295` | `0.2008` |
| SG-TD3 weights previous | `0.001326` | `0.1571` |
| DDQN horizon previous | `0.001152` | `0.2042` |
| Dueling horizon previous | `0.001252` | `0.2198` |

Early first-20 post-warm tracking MAE:

| Method | x24 ethane MAE | T85 MAE |
|---|---:|---:|
| OF-MPC | `0.001387` | `0.1707` |
| SG-TD3 weights gaussian | `0.001428` | `0.1683` |
| DDQN horizon epsilon | `0.001071` | `0.2035` |
| Dueling horizon epsilon | `0.001081` | `0.1858` |

Interpretation:

- SG-TD3 weights improves tail temperature and composition relative to OF-MPC.
- Horizon DDQN improves composition but worsens T85.
- Dueling horizon improves composition relative to OF-MPC but still has worse T85 than OF-MPC in the tail.

The horizon reward gains are therefore not pure tracking wins. They are mostly composition wins with remaining temperature weakness and unsafe episodes.

## Gate And Exploration Diagnostics

The latest weights run is better because the SG gate now lets the policy act.

| Weights run | Window | SG policy selected | SG supervisor selected | Median advantage | Executed log-dispersion |
|---|---|---:|---:|---:|---:|
| Gaussian SG-TD3 | early first 20 post-warm | `27.9%` | `72.1%` | `-2.174` | `0.0268` |
| Gaussian SG-TD3 | tail 20 | `57.9%` | `42.1%` | `2.137` | `0.0814` |
| Previous SG-TD3 | early first 20 post-warm | `2.7%` | `97.3%` | `-700.798` | `0.0003` |
| Previous SG-TD3 | tail 20 | `32.3%` | `67.7%` | `-5.708` | `0.0379` |

This is the mechanism we wanted. Gaussian exploration plus zero advantage margin changed the weights family from near-identity execution into a genuinely active, but still supervised, TD3 weight policy.

The horizon diagnostics tell a different story.

| Horizon run | Tail unique recipes | Tail top recipe | Top fraction | Default `(6, 3)` fraction | Tail switch fraction | Final saved epsilon | Tail loss mean |
|---|---:|---|---:|---:|---:|---:|---:|
| DDQN epsilon | `87` | `(4, 3)` | `4.6%` | `0.2%` | `17.4%` | `0.133` | `14.09` |
| Dueling epsilon | `87` | `(4, 2)` | `14.0%` | `0.5%` | `16.3%` | `0.133` | `6.61` |
| DDQN previous | `86` | `(12, 5)` | `8.1%` | `0.6%` | `22.2%` | `0.000` | `53.48` |
| Dueling previous | `79` | `(9, 2)` | `17.2%` | `10.6%` | `17.7%` | `0.000` | `22.68` |

Epsilon-greedy improved the DQN losses and the rewards, especially for dueling. But the policy is still using almost the whole 87-action grid in the tail. Standard DDQN is especially unconcentrated: its most common tail recipe appears only `4.6%` of the time.

## Why The Epsilon Trace Ends At 0.133

The runner sets `eps_decay_steps = 50000`, but `DQNAgent.take_action` increments `self.steps` only when a live DQN action is requested. Horizon actions are requested at decision intervals, not every plant step.

For this run:

$$ N_{\mathrm{decisions}} \approx \frac{80000 - 4000 - 1200}{4} = 18700. $$

With linear decay:

$$ \epsilon(n) = 0.20 + \min(1,n/50000)(0.02-0.20). $$

At `n = 18700`, this gives:

$$ \epsilon \approx 0.133. $$

That matches the saved trace. So this is not a random logging artifact. It is a schedule-scale mismatch.

For the next horizon run, use:

$$ \mathrm{eps\_decay\_steps} \approx 18000\text{ to }20000. $$

That should make the tail epsilon actually reach `0.02` during the current `80k` plant-step protocol.

## Interpretation

### 1. Weights SG-TD3 Is Working Now

The latest weights result supports the previous diagnosis. The disappointing June 2 weights run was gate-limited and exploration-limited in the executed action, not fundamentally impossible.

The new run:

- Increases tail reward by `+4.04` over the previous weights SG-TD3 run.
- Improves tail T85 MAE from `0.1571` to `0.1545`.
- Improves tail composition MAE from `0.001326` to `0.001216`.
- Raises tail policy selection from `32.3%` to `57.9%`.

The remaining weakness is early release. The worst first-20 post-warm reward falls to `1.63`, lower than the previous `8.35` and OF-MPC `9.07`. It is still nonnegative, so this is not the urgent safety issue.

### 2. Horizon Epsilon-Greedy Helped But Did Not Solve Safety

The latest horizon results are much better than the June 2 diagnosis. Dueling is the most improved horizon run:

- Tail reward improves from `3.99` to `9.29`.
- Final reward improves from `1.56` to `15.22`.
- Tail T85 MAE improves from `0.2198` to `0.2008`.

But both horizon agents still allow bad recipes into the plant. The standard DDQN crash to `-22.86` is too large to accept. The dueling agent has a better tail but more negative episodes.

That is exactly the failure mode a safety-gated DQN should address.

### 3. Do Not Prune The Horizon Range First

The latest horizon tail recipes are concentrated near short horizons, especially `(4, 2)`, `(4, 3)`, and `(5, 2)`. However, standard DDQN also uses higher recipes such as `(13, 10)`, and the current grid is still providing some reward upside.

If we prune the range before adding the gate, we mix two explanations:

- Did safety improve because SG-DQN rejected bad changes?
- Or did safety improve because the bad actions were removed?

For a clean ablation, keep the 87 recipes for the first SG-DQN run. Use the SG-DQN accepted-action logs to decide whether a curated grid should be the second ablation.

## Recommended Next Experiment

### Primary Run: Dueling SG-DQN Horizon

Purpose: preserve the latest dueling horizon upside while removing negative episodes.

Likely files:

- `DuelingDQN/dueling_dqn_agent.py`
- `utils/agent_step_runtime.py`
- `utils/horizon_runner_dueling.py`
- `utils/horizon_safety.py`
- `systems/distillation/notebook_params.py`
- New wrapper file, likely `distillation_RL_assisted_MPC_horizons_dueling_sg_dqn_unified.py`

Core gate:

$$ A_Q(s_k,a_{\mathrm{cand}}) = Q(s_k,a_{\mathrm{cand}}) - Q(s_k,a_{\mathrm{sup}}). $$

Execute the candidate only when:

$$ A_Q(s_k,a_{\mathrm{cand}}) \ge m_Q,\quad d_{\mathrm{hold}} \ge d_{\min},\quad \epsilon_{\mathrm{recent,T85}} \le \epsilon_{\max}. $$

Otherwise execute:

$$ a_k^{\mathrm{exec}} = a_{\mathrm{sup}}. $$

Start with:

| Setting | Initial value |
|---|---:|
| supervisor during release | `(6, 3)` |
| supervisor after release | last accepted recipe |
| fallback if last accepted is unavailable | `(6, 3)` |
| value margin `m_Q` | `0.0` |
| tie behavior | supervisor wins |
| minimum dwell | `2` DQN decisions |
| epsilon schedule | `0.20 -> 0.02` |
| epsilon decay steps | `18000` to `20000` action calls |
| grid | current 87 recipes |

Metrics that must improve:

- Negative post-warm episodes: target `0`.
- Worst first-20 post-warm reward: target at least OF-MPC-safe, ideally above `4.49`.
- Tail reward: target above `9.29` or at least not below `8.5`.
- Tail T85 MAE: target at or below OF-MPC `0.1921`.
- Tail unique recipes: target below `30`.
- Tail switch fraction: target below `0.05`.
- Accepted action fraction: target above `20%` so the gate is not just copying `(6, 3)`.

Failure modes to watch:

- Gate collapses to `(6, 3)` and loses the horizon upside.
- Q-values are overconfident and still accept unsafe recipes.
- Dwell reduces switch count but does not remove bad episodes.
- T85 remains worse than OF-MPC even if reward improves.

### Paired Run: Standard SG-DQN Horizon

Purpose: check whether the gate works better with the standard Q network.

The standard latest run has fewer negative post-warm episodes than dueling, but its worst crash is much deeper. If standard SG-DQN removes the crash and keeps the composition gain, it may become a more robust horizon option than dueling.

Use the same gate, grid, epsilon decay, and logging as the dueling SG-DQN run.

### Secondary Run: Weights SG-TD3 Release Smoothing

Purpose: keep the improved tail reward while lifting the early post-warm minimum.

This is lower priority than SG-DQN because the latest weights run has no negative post-warm episodes. If we tune it, use a small release-smoothing ablation rather than changing the reward:

- Keep Gaussian exploration.
- Keep `advantage_margin = 0.0`.
- Add a 5-subepisode mild policy-selection ramp after critic warm.
- Or require `advantage > 0` plus a small recent-reward guard only during subepisodes `14` to `25`.

Metric to improve: worst first-20 post-warm reward from `1.63` toward OF-MPC `9.07`, while keeping tail reward above `18`.

## Logging To Add For SG-DQN

The SG-DQN result bundle should save:

- `sg_dqn_candidate_action_log`
- `sg_dqn_supervisor_action_log`
- `sg_dqn_executed_action_log`
- `sg_dqn_q_candidate_log`
- `sg_dqn_q_supervisor_log`
- `sg_dqn_advantage_log`
- `sg_dqn_selected_source_log`
- `sg_dqn_rejection_reason_log`
- `sg_dqn_dwell_counter_log`
- `sg_dqn_recent_reward_guard_log`
- `sg_dqn_recent_t85_guard_log`
- `sg_dqn_accept_fraction_by_episode`

Useful source codes:

| Code | Meaning |
|---:|---|
| `0` | warm default |
| `1` | release default |
| `2` | accepted DQN |
| `3` | held previous |
| `4` | rejected by value margin |
| `5` | rejected by dwell |
| `6` | rejected by recent reward or T85 guard |
| `7` | fallback after solver issue |

## Literature Connections

No new external citation is added in this report. The interpretation relies on verified local code and saved bundles.

The method connection is still clear:

- Double DQN is the correct baseline for discrete horizon selection because the action is a finite recipe index.
- Dueling DQN is useful when state value and action advantage should be separated across many similar recipes.
- SG-TD3 showed that a learned controller can keep late upside while avoiding release collapse when a supervisor gate controls execution.
- SG-DQN is the discrete analogue: compare a candidate recipe against a supervisor or held recipe using learned Q-values, then execute only when the advantage and dwell conditions are satisfied.

## Remaining Uncertainty

The current horizon bundles are training rollouts, not frozen evaluation rollouts. Some late recipe churn may be exploration, some may be learned policy instability.

The Q-value scale for SG-DQN is not yet logged. The first SG-DQN run should therefore start with `m_Q = 0.0` and log the advantage distribution before choosing a nonzero margin.

The saved horizon bundles do not contain shadow default-MPC objective comparisons. If SG-DQN still accepts bad recipes, shadow default diagnostics should be turned on before pruning the grid or changing the reward again.

## Conclusion

The next high-value experiment is SG-DQN, not reward retuning and not immediate range reduction.

The current evidence says:

- Weights SG-TD3 with Gaussian exploration is now working and safe enough.
- Horizon epsilon-greedy improved learning but left unsafe episodes.
- The epsilon schedule should be corrected to action-call scale.
- A value-margin plus dwell gate is the missing safety mechanism for horizon control.

Run dueling SG-DQN first, then standard SG-DQN as the paired check. Keep the full grid for that ablation, and only prune the action set after the SG-DQN logs tell us which recipes are actually trusted.
