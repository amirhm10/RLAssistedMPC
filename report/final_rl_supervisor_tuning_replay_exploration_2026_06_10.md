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

## Current Default Snapshot

| Case | Family | Learning rate | Replay | Exploration | Gate margin |
| --- | --- | --- | --- | --- | --- |
| Polymer | horizon SG-DQN | `lr = 1e-4` | 150000, PER 0.5, recent 0.2 | epsilon `0.2 -> 0.02` | 0.0 |
| Polymer | Markov SG-TD3 | actor `1e-4`, critic `1e-4` | 150000, PER 0.5, recent 0.2 | Gaussian `0.2 -> 0.02` | 0.0 |
| Polymer | weights SG-TD3 | actor `1e-4`, critic `1e-4` | 150000, PER 0.5, recent 0.2 | Gaussian `0.2 -> 0.02` | 0.5 |
| Polymer | residual SG-TD3 | actor `1e-4`, critic `1e-4` | 150000, PER 0.5, recent 0.2 | Gaussian `0.2 -> 0.02` | 0.5 |
| Distillation | horizon SG-DQN | `lr = 1e-4` | 40000, PER 0.4, recent 0.3 | epsilon `0.2 -> 0.02` | 0.0 |
| Distillation | Markov SG-TD3 | actor `1e-4`, critic `1e-4` | 40000, PER 0.4, recent 0.3 | param-noise `0.10 -> 0.02` | 0.5 |
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

## Question 4: Should Distillation Exploration Start At 0.05 Instead Of 0.2?

For high-authority continuous distillation agents, yes, exploration should not start at `0.2`. The active defaults already moved in that direction:

- Markov distillation uses parameter noise `0.10 -> 0.02`.
- Residual distillation uses parameter noise `0.10 -> 0.02`.
- Weights distillation uses Gaussian noise `0.15 -> 0.03`.
- Horizon distillation uses epsilon `0.2 -> 0.02`, but horizon actions are lower authority because they choose MPC recipes rather than direct residual inputs.

The stronger question is whether to reduce continuous distillation exploration from `0.10` or `0.15` down to `0.05`.

Recommendation:

- Distillation residual: keep `param_noise_std_start = 0.10` for the default, because the latest residual run is the best current evidence and had no negative post-warm episodes.
- Distillation Markov: run one final safety ablation with `param_noise_std_start = 0.05`, `param_noise_std_end = 0.02` only if you want to prioritize removing the remaining negative post-warm episode over maximizing tail reward.
- Distillation weights: keep `std_start = 0.15`, `std_end = 0.03` unless the combined run shows weight-induced oscillation. The weights action is filtered through MPC and has been stable.
- Distillation horizon: keep `eps_start = 0.2`, `eps_end = 0.02`. Lowering epsilon to `0.05` would likely make the horizon policy collapse too early to the supervisor recipe.
- Polymer continuous agents: keep `std_start = 0.2`, `std_end = 0.02` unless the new combined `z_bound = 0.7` run shows release instability. Polymer has tolerated wider action/range sweeps better than distillation.

If only one final exploration polish is allowed, choose distillation Markov `param_noise_std_start = 0.05` as an ablation, not residual. Residual is currently the distillation winner.

## Main Result Interpretation

The current settings are mostly where they should be for finalization:

- Learning rates are conservative and should stay at `1e-4/1e-4`.
- Separate replay buffers are correct and should stay.
- Distillation continuous exploration should remain softer than polymer, but residual should not be changed away from its current winning setting without a paired ablation.
- Margin scheduling is useful as an idea, but only for carefully targeted release smoothing. A large margin contracted all the way to zero is too aggressive for final distillation defaults.

The strongest final distinction is between systems:

- Polymer can tolerate more Markov authority. The latest standalone Markov range is `z_bound = 0.7`, and new combined runs should inherit it.
- Distillation needs softer live release. The latest residual win came from low first-live policy fraction and parameter noise, not from letting the actor explore aggressively.

## Bugs, Inconsistencies, Or Risks Found

- The latest saved polymer combined run used `markov_z_bound = 0.2`, while the current standalone Markov default is `0.7`. A new combined run is needed before claiming combined evidence under the final Markov range.
- The latest distillation combined output folder has no `input_data.pkl`, so the combined distillation run cannot be audited for replay, source fractions, or learning traces from saved data.
- Some saved Markov bundles do not persist replay size/snapshot fields as consistently as weights and residual bundles.
- Loss magnitudes are not directly comparable between polymer and distillation because reward and Q scales differ. They should be used for instability screening, not as a cross-system performance metric.

## Literature Connections

The TD3 paper motivates the current twin-critic and delayed-actor structure. It specifically targets actor-critic value overestimation by using the minimum of two critics and delaying policy updates, which supports conservative learning-rate choices when the critic also controls a safety gate. Source: [Fujimoto et al., 2018](https://arxiv.org/abs/1802.09477).

The SAC paper emphasizes that deep off-policy RL can suffer high sample complexity and brittle convergence, and motivates entropy/stochasticity as a stabilizing design. This supports treating exploration and learning-rate changes as controlled ablations rather than final broad changes. Source: [Haarnoja et al., 2018](https://arxiv.org/abs/1801.01290).

Prioritized replay supports replaying more informative transitions more often than uniform replay. The repo's PER plus recent plus uniform sampler is consistent with this idea while adding extra emphasis on the current closed-loop regime. Source: [Schaul et al., 2015](https://arxiv.org/abs/1511.05952).

DQN introduced replay memory and target networks for value learning with neural networks, and Double DQN motivates the repo's preference for overestimation-aware value estimates and conservative gates. Sources: [Mnih et al., 2013](https://arxiv.org/abs/1312.5602), [van Hasselt et al., 2015](https://arxiv.org/abs/1509.06461).

Parameter-space noise supports the use of parameter perturbations for temporally coherent exploration. This is especially relevant for distillation residual and Markov agents, where independent stepwise action noise can excite the column. Source: [Plappert et al., 2017](https://arxiv.org/abs/1706.01905).

## Recommended Next Experiments

1. Polymer combined final Markov-range run  
   Purpose: verify that combined polymer works with the standalone Markov `z_bound = 0.7`.  
   File: `RL_assisted_MPC_combined_unified.py`.  
   Change: no code change required if defaults are current.  
   Metrics: tail reward, worst post-warm reward, Markov source fraction, Markov projection fraction, weight/residual policy fractions.  
   Confirming result: tail reward remains near or better than the 20260609 combined run, without a new post-warm collapse.

2. Distillation Markov exploration ablation  
   Purpose: test whether softer parameter noise removes the remaining negative post-warm episode.  
   File: `systems/distillation/notebook_params.py`.  
   Change for ablation only: `markov_td3["param_noise_std_start"] = 0.05`, keep `param_noise_std_end = 0.02`.  
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

## Remaining Uncertainty

The learning-rate conclusion is based on current defaults, saved loss traces, and result behavior, not a formal LR sweep. The replay recommendation is much stronger because the current agent-specific state/action spaces make separate buffers structurally appropriate. The exploration recommendation is strongest for avoiding `0.2` in high-authority distillation continuous agents, but the exact choice between `0.10` and `0.05` remains an empirical tradeoff between early safety and final tail reward.
