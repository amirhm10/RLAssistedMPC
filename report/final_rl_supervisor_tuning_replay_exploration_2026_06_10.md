# Final RL Supervisor Tuning, Replay, And Exploration Recommendations

Date: 2026-06-10  
Cases: polymer CSTR and distillation column  
Scope: active SG combined runners, standalone Markov evidence, and latest saved family reports

## Objective

This note answers four final-tuning questions before freezing the project defaults:

1. Whether the current actor and critic learning rates are the best final choice.
2. What a large supervisor-gate margin that contracts toward zero over 10 episodes would do.
3. Whether combined supervisors use one replay buffer or separate buffers, and which design is preferable.
4. Whether distillation exploration should start near `0.05` instead of `0.2`.

The recommendations below are conservative. They are meant for final project polish, not a new broad hyperparameter search.

## Files Inspected

- `systems/polymer/notebook_params.py`
- `systems/distillation/notebook_params.py`
- `RL_assisted_MPC_combined_unified.py`
- `distillation_RL_assisted_MPC_combined_unified.py`
- `utils/combined_runner.py`
- `utils/agent_step_runtime.py`
- `utils/supervisor_gated_action.py`
- `TD3Agent/agent.py`
- `TD3Agent/replay_buffer.py`
- `TD3Agent/supervisor_gated_agent.py`
- `TD3Agent/supervisor_replay_buffer.py`
- `DQN/dqn_agent.py`
- `DQN/replay_buffer.py`
- `DQN/supervisor_gated_dqn_agent.py`
- `report/distillation_all_runners_latest_analysis_2026_06_09.md`
- `report/polymer_latest_family_runs_2026_05_21.md`
- `report/polymer_sg_td3_weight_residual_latest_2026_06_01.md`
- `Polymer/Results/sg_td3_markov_critic_warm3_ls_else_mpc_shadow_disturb_mismatch/20260610_075904/input_data.pkl`
- `Polymer/Results/combined_disturb_sg__h_sg_dqn_mismatch__markov_sg_td3_mismatch__w_sg_td3_mismatch__r_sg_td3_mismatch_no_rho/20260609_155026/input_data.pkl`
- `Distillation/Results/distillation_residual_sg_td3_critic_warm3_margin05_paramnoise_manual_off_disturb_fluctuation_mismatch_no_rho/20260608_183743/input_data.pkl`
- `Distillation/Results/distillation_markov_sg_td3_critic_warm3_margin05_softparamnoise_ls_else_mpc_shadow_disturb_fluctuation_mismatch/20260608_194745/input_data.pkl`
- `Distillation/Results/distillation_weights_sg_td3_critic_warm3_margin0_sup001_gauss015_003_manual_off_disturb_fluctuation_mismatch/20260608_181133/input_data.pkl`
- `Distillation/Results/distillation_horizon_sg_dqn_critic_warm3_default_ofmpc_eps02_002_disturb_fluctuation_mismatch_np6_11_nc3_11/20260608_180237/input_data.pkl`
- `Distillation/Results/distillation_dueling_horizon_sg_dqn_critic_warm3_default_ofmpc_eps02_002_disturb_fluctuation_mismatch_np6_11_nc3_11/20260608_180736/input_data.pkl`
- `report/scripts/analyze_distillation_replay_state_ranges_20260610.py`
- `report/figures/distillation_replay_state_ranges_20260610/distillation_replay_state_range_summary.csv`
- `report/figures/distillation_replay_state_ranges_20260610/distillation_replay_state_feature_detail_tail20_steady.csv`
- `report/scripts/analyze_final_combined_agent_attribution_20260610.py`
- `report/figures/final_combined_agent_attribution_20260610/final_agent_attribution_summary.csv`
- `report/figures/final_combined_agent_attribution_20260610/final_run_journal_table.csv`
- `report/figures/final_combined_agent_attribution_20260610/final_logging_gap_table.csv`
- `change-reports/2026-06-10_final_param_noise_ablation_defaults.md`

## Current Default Snapshot

| Case | Family | Learning rate | Replay | Exploration | Gate margin |
| --- | --- | --- | --- | --- | --- |
| Polymer | horizon SG-DQN | `lr = 1e-4` | 150000, PER 0.5, recent 0.2 | epsilon `0.2 -> 0.02` | 0.0 |
| Polymer | Markov SG-TD3 | actor `1e-4`, critic `1e-4` | 150000, PER 0.5, recent 0.2 | param-noise `0.10 -> 0.02` | 0.0 |
| Polymer | weights SG-TD3 | actor `1e-4`, critic `1e-4` | 150000, PER 0.5, recent 0.2 | Gaussian `0.2 -> 0.02` | 0.5 |
| Polymer | residual SG-TD3 | actor `1e-4`, critic `1e-4` | 150000, PER 0.5, recent 0.2 | param-noise `0.10 -> 0.02` | 0.5 |
| Distillation | horizon SG-DQN | `lr = 1e-4` | 40000, PER 0.4, recent 0.3 | epsilon `0.2 -> 0.02` | 0.0 |
| Distillation | Markov SG-TD3 | actor `1e-4`, critic `1e-4` | 40000, PER 0.4, recent 0.3 | param-noise `0.05 -> 0.02` | 0.5 |
| Distillation | weights SG-TD3 | actor `1e-4`, critic `1e-4` | 40000, PER 0.4, recent 0.3 | Gaussian `0.15 -> 0.03` | 0.0 |
| Distillation | residual SG-TD3 | actor `1e-4`, critic `1e-4` | 40000, PER 0.4, recent 0.3 | param-noise `0.10 -> 0.02` | 0.5 |

The latest standalone polymer Markov run saved `markov_z_bound = 0.7`. The latest saved polymer combined run still saved `markov_z_bound = 0.2`, so it predates the final standalone Markov authority range. New polymer combined runs inherit the current standalone Markov default, but the old saved combined metrics should not be treated as evidence for `z_bound = 0.7`.

The latest distillation combined output folder `Distillation/Results/distillation_combined_sg_disturb_fluctuation/20260610_001929/` contains figures but no `input_data.pkl`, so I could not audit its replay contents, loss traces, or source-fraction logs.

## Mathematical Interpretation

The inner controller remains offset-free MPC. Each RL agent changes one supervisory variable rather than replacing MPC:

$$ \min_{\Delta U} \sum_{i=1}^{N_p} (y_{k+i}-y_{\mathrm{sp},k+i})^\top Q (y_{k+i}-y_{\mathrm{sp},k+i}) + \sum_{i=0}^{N_c-1} \Delta u_{k+i}^\top R \Delta u_{k+i}. $$

For a continuous SG-TD3 block, the actor proposes a normalized action `a_policy`, and the supervisor gives a safe or nominal action `a_sup`. The gate computes conservative critic scores and executes the actor only when

$$ S(s_k,a_{\mathrm{policy}}) > S(s_k,a_{\mathrm{sup}}) + m_k. $$

Here `m_k` is the `advantage_margin`. A larger margin makes live actor execution rarer. A scheduled margin can be written as

$$ m_q = m_{\mathrm{final}} + (m_0-m_{\mathrm{final}})\max(0,1-q/H_{\mathrm{release}}), $$

where `q` is the number of post-freeze subepisodes since live release and `H_release = 10` if the contraction lasts 10 episodes.

Each combined agent stores its own transition:

$$ \mathcal{D}_i \ni (s^i_k,a^i_k,r_k,s^i_{k+1},d_k). $$

The reward and plant trajectory are shared, but `s^i` and `a^i` differ by agent family. Horizon actions are discrete recipes, Markov actions are correction coordinates, weight actions are penalty multipliers, and residual actions are input corrections.

## Question 1: Are The Current Learning Rates Best?

Short answer: they are not proven globally best, but they are the best defensible final default from the evidence in this repo.

The active default `actor_lr = critic_lr = 1e-4` is conservative. That matters because the current networks are large, the value scale differs strongly between polymer and distillation, and the gate uses the critic directly to decide whether the actor is allowed to control the plant. A higher learning rate can make the critic adapt faster, but it also makes the gate less trustworthy during the first live-release window.

The latest saved loss traces do not show a reason to increase the learning rates as a final-touch change:

| Bundle | Main observation |
| --- | --- |
| Polymer combined 20260609 | TD3 critic losses were finite and decayed to small tail means for weights and residual, while actor losses grew because the Q scale changed during learning. This does not by itself imply the actor LR is too small. |
| Polymer standalone Markov 20260610 | Critic losses remained finite under the new `z_bound = 0.7`; the run is recent positive evidence for leaving Markov `1e-4/1e-4` alone. |
| Distillation residual 20260608 | Tail reward was the best latest distillation result, with no negative post-warm episodes, under `1e-4/1e-4`. |
| Distillation Markov 20260608 | Markov had one negative post-warm episode, but the report suggests the problem may be supervisor/candidate scoring around the column state, not an obvious LR failure. |

Recommendation:

- Keep `actor_lr = critic_lr = 1e-4` for both polymer and distillation final defaults.
- Do not increase either rate at this stage.
- If a single safety ablation is still desired, test distillation Markov or residual with `actor_lr = 5e-5`, `critic_lr = 1e-4`. This is a safer ablation than making both rates larger, because it slows the policy while preserving critic adaptation.
- Do not change polymer learning rates unless the new combined `z_bound = 0.7` run shows unstable critic scores or strong policy oscillation.

## Question 2: Large Margin Then Contract To Zero Over 10 Episodes

A large margin followed by contraction is a release schedule. It would probably reduce the first-live shock, but it can also delay learning or create a second shock when the margin reaches zero.

Expected behavior:

- Early release: a large margin forces the supervisor to execute unless the critic is extremely confident in the actor. This protects the plant while replay fills with post-warm states.
- Middle release: as the margin decreases, actor executions become more frequent. This creates a smoother transition than an immediate margin drop.
- End of schedule: if the final margin is zero, the gate becomes permissive. For high-authority distillation residual and Markov actions, this can recover tail performance but may reintroduce fragile episodes.

Recommended schedules:

| Case | Agent family | Recommended final margin schedule |
| --- | --- | --- |
| Polymer Markov | SG-TD3 | Do not add a large margin now. The latest standalone result is explicitly `z_bound = 0.7`, `advantage_margin = 0.0`. |
| Polymer weights/residual | SG-TD3 | Optional gentle schedule `0.5 -> 0.0` over 10 episodes if combined is too conservative. Keep it as an ablation, not the default. |
| Distillation Markov | SG-TD3 | Prefer `1.0 -> 0.5` over 10 episodes, not `1.0 -> 0.0`, because latest Markov still had one negative post-warm episode at margin 0.5. |
| Distillation residual | SG-TD3 | Keep `0.5` fixed as the final default unless seed replication shows the actor is too conservative. The latest residual result is already the current winner. |
| Distillation weights | SG-TD3 | Keep margin 0.0. The action changes MPC weights rather than direct input residuals, and the latest batch was stable. |
| Horizon DQN | SG-DQN | Keep margin 0.0. Exploration is through discrete recipe selection, and the SG supervisor already anchors the OF-MPC recipe. |

I would not make a margin-to-zero schedule the default for distillation residual or Markov at the very end. If you want the schedule, the safer final-polish version is margin-to-current-default, not margin-to-zero.

## Question 3: Replay Situation In Combined Runs

The combined runner uses separate replay buffers, one per active agent. This is the correct design for the current architecture.

Evidence from code:

- `RL_assisted_MPC_combined_unified.py` creates separate objects for `horizon_agent`, `markov_agent`, `weights_agent`, and `residual_agent`.
- `distillation_RL_assisted_MPC_combined_unified.py` follows the same pattern.
- `utils/combined_runner.py` retrieves these as separate agents and calls replay/train helpers separately for each block.
- `TD3Agent/agent.py` gives each TD3 or SG-TD3 object its own `PERRecentReplayBuffer`.
- `DQN/dqn_agent.py` gives each DQN or SG-DQN object its own `PERRecentReplayBuffer`.

The buffers are not a single shared memory. They are synchronized by the common plant rollout and reward, but each stores its own state/action representation.

This is best for the present project because:

- The action dimensions and meanings are incompatible across agents.
- A shared buffer would confuse credit assignment unless you redesign the method around a centralized critic or joint action.
- Separate buffers let each agent learn from the same closed-loop episode while retaining its own Markov state, action, supervised metadata, and replay priorities.
- The current hybrid sampler already prevents old warm-start data from dominating: each batch is part prioritized, part recent, and part uniform.

The only reason to use a shared buffer would be a different method: a centralized multi-agent critic with joint action

$$ a_k^{\mathrm{joint}} = [a^h_k,a^M_k,a^w_k,a^r_k], $$

and a joint state. That is a larger research contribution, not a final-touch cleanup.

Recommendation:

- Keep separate buffers for final runs.
- Keep distillation at the active replay profile: 40000 transitions, PER 0.4, recent 0.3, recent window multiplier 10.
- Keep polymer at 150000 transitions, PER 0.5, recent 0.2, recent window multiplier 5.
- For fair combined diagnostics, save buffer sizes and replay snapshots consistently for all blocks. The latest distillation Markov report noted that Markov replay size was not saved in the same way as other bundles.

## Question 3b: Are Distillation Replay States Too Narrow To Inform RL?

Short answer: the full replay buffers are not too narrow, but the steady-state tail slices are narrow for weights, horizon, and dueling horizon. That is expected and not automatically bad. The steady data mostly teaches the actor and critic to maintain the safe near-setpoint behavior. The informative transient data is still present in the full replay snapshot.

The latest distillation saved bundles give direct access to replay states for weights, residual, horizon, and dueling horizon through `replay_buffer_snapshot["states"]`. Each snapshot is `40000 x 15`, which is the full replay capacity. The 15 dimensions are:

- 11 base RL state features from the augmented observer state, setpoint, and previous input.
- 2 transformed innovation features.
- 2 transformed tracking-error features.

The mismatch features use the `signed_log` transform:

$$ z_{\mathrm{mis}} = \mathrm{sign}(e_{\mathrm{raw}})\log(1+|e_{\mathrm{raw}}|). $$

That means small transformed values really do indicate small band-normalized innovation or tracking error. For Markov, the latest standalone bundle does not save an exact replay snapshot, but it does save `rl_state_log` with shape `80000 x 25` and `rl_replay_pushed_log`; 99.5 percent of those logged Markov states were pushed. This is useful evidence, but it is not the exact replay-buffer export.

The table below compares the full replay snapshot with the last 100 steps of each episode over the final 20 episodes. The latter is the most steady-state-heavy slice.

| Run | Source | Segment | State dim | Median feature range | Max feature range | Rounded-3 unique frac | Max tracking < 0.05 | Max tracking < 0.10 | Policy source frac |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| weights | replay snapshot | all | 15 | 0.567 | 7.442 | 0.868 | 27.3% | 32.9% | 0.596 |
| weights | replay snapshot | tail steady | 15 | 0.029 | 0.106 | 0.556 | 100.0% | 100.0% | 0.481 |
| residual | replay snapshot | all | 15 | 0.546 | 7.052 | 0.915 | 42.6% | 60.7% | 0.386 |
| residual | replay snapshot | tail steady | 15 | 0.074 | 0.905 | 0.999 | 50.8% | 83.5% | 0.349 |
| Markov | state log | all | 25 | 0.787 | 6.000 | 0.871 | NA | NA | NA |
| Markov | state log | tail steady | 25 | 0.015 | 0.737 | 0.906 | NA | NA | NA |
| horizon | replay snapshot | all | 15 | 0.571 | 7.704 | 0.847 | 37.0% | 45.0% | 0.157 |
| horizon | replay snapshot | tail steady | 15 | 0.029 | 0.292 | 0.794 | 91.4% | 98.2% | 0.211 |
| dueling | replay snapshot | all | 15 | 0.574 | 7.795 | 0.867 | 34.8% | 45.8% | 0.127 |
| dueling | replay snapshot | tail steady | 15 | 0.029 | 0.210 | 0.743 | 93.4% | 98.8% | 0.131 |

Interpretation:

- The full replay buffers are not collapsed. Full-buffer median feature ranges are about `0.55` to `0.57` for weights, residual, horizon, and dueling horizon, and rounded-to-0.001 unique-state fractions are about `0.85` to `0.91`.
- The steady-state tail slice is narrow for weights, horizon, and dueling horizon. In that slice, more than `91%` of rows have both tracking features below `0.05` or nearly below it. This confirms your concern for the steady-state portion.
- Residual is different. In the residual tail-steady slice, only `50.8%` of rows have max transformed tracking below `0.05`, and the T85 tracking feature has a wide tail range. This means residual still sees meaningful near-steady variation, especially in the temperature channel.
- Markov's exact replay cannot be audited from the latest saved bundle, but its logged state stream is not globally collapsed. Its tail-steady median range is small, yet the rounded unique fraction is still about `0.906`, suggesting continuous variation remains in some Markov-specific features.

For residual, the tail-steady feature detail shows why it is still informative:

| Feature | Range | Std | q05 | Median | q95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `u_reflux` | 0.108 | 0.030 | -0.277 | -0.234 | -0.187 |
| `u_reboiler` | 0.012 | 0.003 | 0.064 | 0.068 | 0.072 |
| `innov_x24` | 0.099 | 0.008 | -0.012 | -0.000 | 0.014 |
| `innov_T85` | 0.131 | 0.012 | -0.019 | -0.000 | 0.018 |
| `track_x24` | 0.122 | 0.013 | -0.012 | 0.007 | 0.030 |
| `track_T85` | 0.905 | 0.076 | -0.146 | -0.021 | 0.096 |

The replay concern is therefore not "there is no informative state." The better diagnosis is:

- Whole-buffer replay still contains enough transient and off-setpoint information.
- The final steady-state slice is low-error and low-innovation for the low-authority agents, so it mostly trains maintenance and safe non-intervention.
- If the recent sampler overemphasizes only the last near-steady regime, it could slow learning about setpoint-change transients. The current hybrid sampler reduces that risk by mixing PER, recent-window samples, and uniform samples.

Final recommendation:

- Keep the current replay design.
- Add a final diagnostic to future saved bundles: percent of replay rows with `max(abs(tracking_features)) < 0.05`, percent with `< 0.10`, and per-feature tail-steady ranges.
- If a future distillation agent becomes too conservative or fails to learn transients, do not first enlarge exploration. First try phase-stratified replay, for example forcing a minimum fraction of minibatch samples from non-steady rows where `max(abs(tracking_features)) >= 0.05`.
- Save exact replay snapshots for Markov and distillation combined. That is now the biggest evidence gap, not the state range of the saved residual/weights/horizon buffers.

## Question 4: Should Distillation Exploration Start At 0.05 Instead Of 0.2?

For high-authority continuous distillation agents, yes, exploration should not start at `0.2`. The active defaults already moved in that direction:

- Markov distillation now uses parameter noise `0.05 -> 0.02`.
- Residual distillation uses parameter noise `0.10 -> 0.02`.
- Weights distillation uses Gaussian noise `0.15 -> 0.03`.
- Horizon distillation uses epsilon `0.2 -> 0.02`, but horizon actions are lower authority because they choose MPC recipes rather than direct residual inputs.
- Polymer Markov and residual now use parameter noise `0.10 -> 0.02`; polymer weights remains Gaussian `0.2 -> 0.02`.

The stronger question was whether to reduce continuous distillation exploration from `0.10` or `0.15` down to `0.05`. The final ablation default now applies that reduction to Markov only.

Recommendation:

- Distillation residual: keep `param_noise_std_start = 0.10` for the default, because the latest residual run is the best current evidence and had no negative post-warm episodes.
- Distillation Markov: use the final safety ablation default `param_noise_std_start = 0.05`, `param_noise_std_end = 0.02`, then evaluate whether the negative post-warm episode disappears without a large tail-reward loss.
- Distillation weights: keep `std_start = 0.15`, `std_end = 0.03` unless the combined run shows weight-induced oscillation. The weights action is filtered through MPC and has been stable.
- Distillation horizon: keep `eps_start = 0.2`, `eps_end = 0.02`. Lowering epsilon to `0.05` would likely make the horizon policy collapse too early to the supervisor recipe.
- Polymer Markov and residual: use final parameter noise `0.10 -> 0.02`. This keeps exploration temporally coherent while avoiding the stepwise actuator jitter of Gaussian action noise.
- Polymer weights: keep Gaussian `0.2 -> 0.02`, because that action changes MPC penalties rather than directly changing input moves or model corrections.

If only one final exploration polish is allowed, run the distillation Markov `param_noise_std_start = 0.05` ablation first. Residual remains the stronger distillation standalone family, but the Markov safety question is more targeted.

## Question 5: How Can We Distinguish Each Agent's Success In Combined Runs?

The most important distinction is diagnostic versus causal attribution.

The current combined logs can say whether an agent was active, trusted by its gate, safe, and learning. They cannot by themselves prove that the closed-loop improvement was caused by that agent, because all active agents see the same plant trajectory and shared reward. True causal attribution requires controlled ablations.

For a combined run with active agent set `A`, the clean leave-one-agent-out contribution is:

$$ C_i = J(A) - J(A \setminus \{i\}), $$

where `J` should be a fixed evaluation metric such as rescored tail reward, IAE, RMSE, or a safety-weighted score. If there are enough reruns, a Shapley-style appendix metric can average each agent's marginal contribution over many coalitions:

$$ \phi_i = \sum_{S \subseteq A \setminus \{i\}} \frac{|S|!(|A|-|S|-1)!}{|A|!}\left[J(S \cup \{i\}) - J(S)\right]. $$

For the current paper, the practical answer is to report two levels:

| Level | What it supports | Metric examples |
| --- | --- | --- |
| Diagnostic attribution | Whether the agent behaved usefully inside one combined rollout | policy fraction, gate advantage, fallback fraction, action authority, safety burden, replay size |
| Causal attribution | Whether the agent improves the closed loop | all-agents versus leave-one-out tail reward, IAE, RMSE, input movement, constraint violations |

The polymer combined bundle is already rich enough for diagnostic attribution. The latest saved distillation combined folder is still missing `input_data.pkl`, so the same audit cannot yet be performed for distillation combined.

| Plant | Agent | Tail policy frac | Final exec policy frac | Median tail advantage | Tail authority diagnostic | Tail safety burden | Replay rows | Interpretation |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| polymer | horizon | 0.208 | NA | 0.078 | 3.377 | NA | 150000 | Horizon is active in recipe selection; use recipe entropy and heatmaps rather than input authority. |
| polymer | Markov | 0.529 | 0.137 | 0.006 | 0.457 | 0.000 | 150000 | The SG gate often prefers policy, but the final Markov execution source is lower; report both. |
| polymer | weights | 0.046 | NA | -0.069 | 0.022 | NA | 150000 | Mostly identity-supervisor behavior; useful if it improves reward while staying low-authority. |
| polymer | residual | 0.049 | NA | 0.0005 | 0.062 | 1.000 | 150000 | Low policy fraction and high projection burden mean residual is constrained; inspect raw versus executed residual before claiming it helped. |
| distillation | combined agents | NA | NA | NA | NA | NA | NA | Missing combined `input_data.pkl`; no auditable attribution yet. |

The agent-specific interpretation should be:

- Horizon: report recipe distribution, tail recipe entropy, OF-MPC recipe fraction, and a tail heatmap over `(N_p,N_c)`.
- Markov: report requested, executed, and LS `z`; prediction score; nominal cost margin; gain drift; final execution source; and projection/fallback fractions.
- Weights: report distance from identity, separate movement of Q and R multipliers, and whether tracking improvement comes with increased input movement.
- Residual: report raw versus executed residual, applied input versus MPC-base input, projection/deadband/saturation, and whether residuals are used during transients or only near steady state.

The generated attribution artifacts are:

- `report/figures/final_combined_agent_attribution_20260610/final_agent_attribution_summary.csv`
- `report/figures/final_combined_agent_attribution_20260610/final_run_journal_table.csv`
- `report/figures/final_combined_agent_attribution_20260610/final_logging_gap_table.csv`
- `report/figures/final_combined_agent_attribution_20260610/fig_final_combined_source_fractions.png`
- `report/figures/final_combined_agent_attribution_20260610/fig_final_combined_gate_advantages.png`
- `report/figures/final_combined_agent_attribution_20260610/fig_final_combined_action_authority.png`
- `report/figures/final_combined_agent_attribution_20260610/fig_final_tail_reward_delta_heatmap.png`

The run table uses saved `avg_rewards` from each bundle, not a common rescoring pass. For final journal numbers, all methods should be rescored with one frozen reward/metric script before making ranking claims.

## Question 6: What Should Be Logged And Plotted For The Journal Paper?

For a journal paper, the evidence package should make three things transparent: closed-loop performance, safety envelope, and learning/attribution mechanism.

Main tables:

| Table | Purpose | Required columns |
| --- | --- | --- |
| Final run summary | Compare OF-MPC, horizon, Markov, weights, residual, and combined for both plants | tail reward, IAE, RMSE, max error, worst post-warm episode, negative post-warm count, input total variation |
| Combined attribution | Explain which block did what | tail policy fraction, final execution fraction, median gate advantage, action authority, safety burden, replay rows |
| Replay/state coverage | Show RL saw informative data | replay size, state dimension, median range, max range, unique fraction, steady-state low-error fraction |
| Safety and constraints | Prove improvement did not come from unsafe inputs | saturation fraction, projection fraction, fallback fraction, solver failures, constraint violations |
| Final config | Make the results reproducible | plant, disturbance profile, seed, run mode, horizons, replay settings, exploration schedule, gate margin, action bounds |

Main figures:

- Tracking and input overlays: OF-MPC, best standalone family, and combined on the same axes for each plant.
- Reward histories with shaded warm-start, critic-only, and live-release windows.
- Per-agent policy/source fraction over subepisodes for combined runs.
- Horizon recipe heatmap over `(N_p,N_c)`.
- Markov requested, executed, and LS `z` panels with bounds.
- Weights multiplier panel showing Q and R movement around identity.
- Residual raw versus executed residual and applied-input versus MPC-base input.
- Replay state-range or PCA/range diagnostic, especially for steady-state-heavy distillation segments.
- One summary heatmap across plant, method family, and normalized metric.

Logging priorities before journal freeze:

| Priority | Add or verify | Why it matters |
| --- | --- | --- |
| P0 | Always save `input_data.pkl` for distillation combined | Without this, combined attribution, replay, losses, and source fractions are not auditable. |
| P0 | Save `episode_metrics.csv` for every run | Avoids re-parsing pickles and gives direct reward, IAE, RMSE, max error, input movement, saturation, and source fractions. |
| P0 | Save `agent_attribution_summary.csv` for every combined run | Gives one table for horizon, Markov, weights, and residual behavior. |
| P1 | Save reward component breakdowns | Prevents reward improvement from hiding tracking or move-suppression regressions. |
| P1 | Save exact replay snapshots or compact replay audits for every active agent | Supports state-coverage and replay-bias claims. |
| P1 | Save provenance fields | Include git commit, config snapshot, baseline path, disturbance profile, seed, timestamp, plant, and model identifiers. |

The final paper should avoid using reward alone as the headline metric. A stronger table is a normalized scorecard with reward, tracking, input movement, safety, and attribution columns. Reward should still be reported, but IAE/RMSE/max error and input total variation are easier for control readers to interpret.

## Main Result Interpretation

The current settings are mostly where they should be for finalization:

- Learning rates are conservative and should stay at `1e-4/1e-4`.
- Separate replay buffers are correct and should stay.
- Distillation continuous exploration should remain softer than polymer, with the final Markov ablation now at `0.05 -> 0.02` and residual kept at `0.10 -> 0.02`.
- Margin scheduling is useful as an idea, but only for carefully targeted release smoothing. A large margin contracted all the way to zero is too aggressive for final distillation defaults.
- Combined-run attribution should be reported diagnostically unless leave-one-agent-out or coalition reruns are available.

The strongest final distinction is between systems:

- Polymer can tolerate more Markov authority. The latest standalone Markov range is `z_bound = 0.7`, and new combined runs should inherit it.
- Distillation needs softer live release. The latest residual win came from low first-live policy fraction and parameter noise, not from letting the actor explore aggressively.

## Bugs, Inconsistencies, Or Risks Found

- The latest saved polymer combined run used `markov_z_bound = 0.2`, while the current standalone Markov default is `0.7`. A new combined run is needed before claiming combined evidence under the final Markov range.
- The latest distillation combined output folder has no `input_data.pkl`, so the combined distillation run cannot be audited for replay, source fractions, or learning traces from saved data.
- Some saved Markov bundles do not persist replay size/snapshot fields as consistently as weights and residual bundles.
- The latest distillation Markov bundle does not save an exact replay buffer snapshot. It saves `rl_state_log`, which is useful but not equivalent to the exact replay export.
- Loss magnitudes are not directly comparable between polymer and distillation because reward and Q scales differ. They should be used for instability screening, not as a cross-system performance metric.

## Literature Connections

The TD3 paper motivates the current twin-critic and delayed-actor structure. It specifically targets actor-critic value overestimation by using the minimum of two critics and delaying policy updates, which supports conservative learning-rate choices when the critic also controls a safety gate. Source: [Fujimoto et al., 2018](https://arxiv.org/abs/1802.09477).

The SAC paper emphasizes that deep off-policy RL can suffer high sample complexity and brittle convergence, and motivates entropy/stochasticity as a stabilizing design. This supports treating exploration and learning-rate changes as controlled ablations rather than final broad changes. Source: [Haarnoja et al., 2018](https://arxiv.org/abs/1801.01290).

Prioritized replay supports replaying more informative transitions more often than uniform replay. The repo's PER plus recent plus uniform sampler is consistent with this idea while adding extra emphasis on the current closed-loop regime. Source: [Schaul et al., 2015](https://arxiv.org/abs/1511.05952).

DQN introduced replay memory and target networks for value learning with neural networks, and Double DQN motivates the repo's preference for overestimation-aware value estimates and conservative gates. Sources: [Mnih et al., 2013](https://arxiv.org/abs/1312.5602), [van Hasselt et al., 2015](https://arxiv.org/abs/1509.06461).

Parameter-space noise supports the use of parameter perturbations for temporally coherent exploration. This is especially relevant for distillation residual and Markov agents, where independent stepwise action noise can excite the column. Source: [Plappert et al., 2017](https://arxiv.org/abs/1706.01905).

COMA motivates the distinction between shared reward and individual-agent credit by using a counterfactual baseline for multi-agent credit assignment. This supports treating current combined logs as diagnostic attribution and leave-one-agent-out reruns as causal attribution. Source: [Foerster et al., 2017](https://arxiv.org/abs/1705.08926).

VDN and QMIX motivate value decomposition as a principled way to reason about cooperative agents under a shared team objective. The current project does not implement VDN or QMIX, but their framing supports reporting horizon, Markov, weights, and residual contributions separately rather than collapsing everything into one reward curve. Sources: [Sunehag et al., 2017](https://arxiv.org/abs/1706.05296), [Rashid et al., 2018](https://arxiv.org/abs/1803.11485).

Shapley counterfactual credits provide a coalition-based way to assign multi-agent contribution. For this repo, Shapley-style analysis should be appendix material only if enough coalition reruns exist; otherwise use leave-one-agent-out metrics. Source: [Li et al., 2021](https://arxiv.org/abs/2106.00285).

Empirical RL reporting papers emphasize that single-run curves are fragile and should be supported by explicit metrics, uncertainty, and reproducible evaluation choices. This supports adding final CSV metric tables and common rescoring scripts before journal submission. Sources: [Patterson et al., 2023](https://arxiv.org/abs/2304.01315), [Agarwal et al., 2021](https://arxiv.org/abs/2108.13264), [Colas et al., 2019](https://arxiv.org/abs/1904.06979).

Safe RL with MPC literature supports the paper framing used here: RL proposes adaptation inside an MPC/safety envelope, while MPC or chance-constrained MPC provides the executable safety structure. Sources: [Koller et al., 2019](https://arxiv.org/abs/1906.12189), [Pfrommer et al., 2021](https://arxiv.org/abs/2112.13941).

## Recommended Next Experiments

1. Polymer combined final Markov-range run  
   Purpose: verify that combined polymer works with the standalone Markov `z_bound = 0.7`.  
   File: `RL_assisted_MPC_combined_unified.py`.  
   Change: no code change required if defaults are current.  
   Metrics: tail reward, worst post-warm reward, Markov source fraction, Markov projection fraction, weight/residual policy fractions.  
   Confirming result: tail reward remains near or better than the 20260609 combined run, without a new post-warm collapse.

2. Distillation Markov final exploration run
   Purpose: test whether the now-active softer parameter noise removes the remaining negative post-warm episode.
   File: `systems/distillation/notebook_params.py`.  
   Change: no further code change required; current default is `param_noise_std_start = 0.05`, `param_noise_std_end = 0.02`.
   Metrics: negative post-warm episodes, worst post-warm reward, tail reward, T85 MAE, x24 MAE, tail policy fraction.  
   Confirming result: negative post-warm episodes drop to zero with less than about 10 percent tail reward loss.

3. Optional distillation margin schedule  
   Purpose: smooth Markov release without permanently blocking the actor.  
   File: `TD3Agent/supervisor_gated_agent.py` or runner-level gate config plumbing if implemented later.  
   Change: use `m = 1.0 -> 0.5` over 10 post-freeze episodes for Markov only.  
   Metrics: first-live policy fraction, worst post-warm reward, tail reward.  
   Confirming result: first-live shock improves without reducing tail policy fraction below the current useful range.

4. Save full distillation combined bundles  
   Purpose: make final combined evidence auditable.  
   File: plotting/save path used by `distillation_RL_assisted_MPC_combined_unified.py` and `utils/plotting_core.py` if needed.  
   Change: ensure the distillation combined timestamp folder stores `input_data.pkl`.  
   Metrics: existence of replay/loss/source logs in the saved bundle.  
   Confirming result: the next combined folder can be loaded and audited like standalone residual/Markov.

5. Add replay state-range diagnostics to final saved bundles
   Purpose: confirm that RL is not learning only from near-identical steady-state rows.
   File: plotting/save or analysis layer around `replay_buffer_snapshot`.
   Change: save per-feature range, standard deviation, and phase fractions for all replay buffers.
   Metrics: full-buffer feature ranges, tail-steady feature ranges, percent of rows with max transformed tracking below `0.05` and `0.10`.
   Confirming result: full buffers retain broad transient variation while steady-state rows remain interpretable as maintenance data.

6. Combined leave-one-agent-out attribution
   Purpose: separate diagnostic attribution from causal contribution.
   File: combined runner configs for polymer and distillation.
   Change: run all-agents, no-horizon, no-Markov, no-weights, and no-residual under identical seeds and disturbances.
   Metrics: delta tail reward, delta IAE, delta RMSE, delta input total variation, and safety/fallback changes.
   Confirming result: each claimed useful agent has a positive or interpretable marginal contribution rather than only a high source fraction.

7. Journal rescoring and figure freeze
   Purpose: make tables paper-safe instead of reward-version dependent.
   File: new or extended report analysis script.
   Change: rescore all final canonical bundles with one frozen metric helper and emit `episode_metrics.csv`, `agent_attribution_summary.csv`, and final figure panels.
   Metrics: table completeness, zero missing required fields for final main-text runs, and reproducible figure paths.
   Confirming result: final paper figures can be regenerated from saved bundles without launching Aspen or rerunning polymer simulations.

## Remaining Uncertainty

The learning-rate conclusion is based on current defaults, saved loss traces, and result behavior, not a formal LR sweep. The replay recommendation is much stronger because the current agent-specific state/action spaces make separate buffers structurally appropriate. The state-range audit shows that full distillation replay buffers are not collapsed, but it does not prove optimal replay composition. The combined-agent attribution table is diagnostic rather than causal until leave-one-agent-out or coalition reruns are available. The exact choice between continuous exploration starts of `0.10` and `0.05` remains an empirical tradeoff between early safety and final tail reward.
