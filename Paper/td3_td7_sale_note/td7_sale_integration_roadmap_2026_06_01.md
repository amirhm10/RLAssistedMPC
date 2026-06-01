# TD7/SALE Integration Roadmap For RL-Assisted MPC

Date: 2026-06-01

## Objective

This report aligns the TD7/SALE paper, Scott Fujimoto's official TD7 implementation, and the current RL-assisted MPC codebase before any TD7 code is written. The purpose is to make the intended implementation concrete enough that a future coding pass can add TD7 without changing the meaning of the current TD3 baselines.

The planned implementation is a faithful TD7 port for the continuous-control TD3 families in this repository. It should not replace TD3. It should add a parallel `TD7Agent` implementation that can be selected with `agent_kind = "td7"` and a new `td7_agent` config block.

## Files And Sources Inspected

Local paper folder:

- `Paper/td3_td7_sale_note/TD7 Paper.pdf`
- `Paper/td3_td7_sale_note/TD7 Paper.md`
- `Paper/td3_td7_sale_note/TD7 Paper.local.md`
- `Paper/td3_td7_sale_note/td3_td7_sale_process_control_note.md`
- `Paper/td3_td7_sale_note/figures/td3_td7_conceptual_support.png`
- `Paper/td3_td7_sale_note/figures/reward_shapes_near_setpoint.png`
- `Paper/td3_td7_sale_note/figures/action_equivalence_regularization.png`
- `Paper/td3_td7_sale_note/figures/sale_sensitivity_gate.png`

Current repo implementation:

- `TD3Agent/agent.py`
- `TD3Agent/actor.py`
- `TD3Agent/critic.py`
- `TD3Agent/replay_buffer.py`
- `utils/agent_step_runtime.py`
- `utils/behavioral_cloning.py`
- `utils/markov_runner.py`
- `utils/matrix_runner.py`
- `utils/structured_matrix_runner.py`
- `utils/weights_runner.py`
- `utils/residual_runner.py`
- `utils/combined_runner.py`
- `systems/distillation/notebook_params.py`
- `distillation_RL_assisted_MPC_markov_td3_only_no_safeguard_current_reward_unified.py`
- `distillation_RL_assisted_MPC_weights_unified.py`
- `distillation_RL_assisted_MPC_residual_unified.py`
- `report/distillation_post_reward_no_probation_5runner_analysis_2026_05_31.md`

Verified external sources:

- Fujimoto, Chang, Smith, Gu, Precup, and Meger, "For SALE: State-Action Representation Learning for Deep Reinforcement Learning", arXiv:2306.02451, NeurIPS 2023. URL: https://arxiv.org/abs/2306.02451
- Official TD7 implementation by Scott Fujimoto. URL: https://github.com/sfujim/TD7

## Figure Audit

The four local figures are useful as conceptual scaffolding, but they should not all be treated as publication-ready evidence.

- `td3_td7_conceptual_support.png` is a conceptual summary of what TD7 adds over TD3. It is useful for internal discussion.
- `action_equivalence_regularization.png` is a useful conceptual plot for explaining why a process-control actor needs a tie-breaker near the setpoint.
- `reward_shapes_near_setpoint.png` is useful conceptually, but it visibly contains unrelated UI overlay artifacts. Regenerate it before using it in a paper, slide deck, or report figure set.
- `sale_sensitivity_gate.png` is useful conceptually, but it also visibly contains unrelated UI overlay artifacts. Regenerate it before publication use.

None of these figures is experimental evidence. They should be labeled as conceptual if reused.

## TD7 And SALE Method

TD7 is best understood as TD3 plus four implementation additions:

- State-action learned embeddings, called SALE.
- Loss-adjusted prioritized replay, called LAP.
- Value target clipping using a tracked target range.
- Policy checkpoints for more stable evaluation.

The main SALE idea is to learn a transition-aware representation from every sampled transition. Let the RL state be `s_k`, the normalized continuous action be `a_k`, the next state be `s_{k+1}`, and the reward be `r_k`.

TD7 defines a state encoder:

$$ z_s = f_{\psi}(s). $$

It also defines a state-action encoder:

$$ z_{sa} = g_{\psi}(z_s, a). $$

The SALE encoder loss uses the next-state embedding as the learning target:

$$ \mathcal{L}_{\mathrm{enc}}(\psi) = \mathbb{E}_{(s,a,s') \sim \mathcal{D}} [ \| g_{\psi}(f_{\psi}(s), a) - \mathrm{stopgrad}(f_{\psi}(s')) \|_2^2 ]. $$

The encoder is decoupled from the actor and critic losses. Actor and critic gradients should not update `f` or `g`. This is important because the paper reports that the decoupled representation is part of the successful SALE design.

TD7 normalizes the state embedding and learned input features with AvgL1Norm:

$$ \mathrm{AvgL1Norm}(x) = \frac{x}{\max(\mathrm{mean}(|x|), \epsilon)}. $$

The actor then uses the raw state and state embedding:

$$ a_{\pi} = \pi_{\theta}(s, z_s). $$

The critic uses the raw state-action pair and both embeddings:

$$ Q_{\phi_i} = Q_{\phi_i}(s, a, z_s, z_{sa}), \quad i \in \{1,2\}. $$

The target action remains TD3-like:

$$ \tilde{a}' = \mathrm{clip}(\pi_{\bar{\theta}}(s', \bar{z}'_s) + \epsilon, -1, 1), \quad \epsilon \sim \mathrm{clip}(\mathcal{N}(0,\sigma^2), -c, c). $$

The target value uses the minimum target critic and value clipping:

$$ y = r + \gamma (1-d) \, \mathrm{clip}(\min_i Q_{\bar{\phi}_i}(s', \tilde{a}', \bar{z}'_s, \bar{z}'_{sa}), Q_{\min}, Q_{\max}). $$

For online RL, the actor loss is the deterministic policy-gradient objective:

$$ \mathcal{L}_{\pi} = -\mathbb{E}[Q(s,\pi(s,z_s),z_s,z_{sa,\pi})]. $$

For offline or imitation-regularized use, TD7 adds a behavior cloning penalty. In this repository, that should be connected to the existing `bc_context` mechanism:

$$ \mathcal{L}_{\pi,\mathrm{BC}} = -\mathbb{E}[Q] + \lambda_{\mathrm{BC}}\mathbb{E}[\|\pi(s,z_s)-a_{\mathrm{target}}\|_2^2]. $$

The process-control interpretation is direct. SALE gives the critic and actor a learned representation of how an action changes the local closed-loop state. It does not create offset-free control by itself. Offset-free behavior still depends on the observer, state features, reward geometry, safety logic, and whether the replay buffer stores the action actually applied to the plant.

## Official Fujimoto TD7 Implementation Map

The official `sfujim/TD7` repo is intentionally compact. It contains `TD7.py`, `buffer.py`, and `main.py`. The relevant implementation choices for this repository are:

- `Hyperparameters` holds TD7 defaults such as `target_update_rate = 250`, `policy_freq = 2`, `target_policy_noise = 0.2`, `noise_clip = 0.5`, `alpha = 0.4`, `min_priority = 1`, and `lmbda = 0.1` for offline BC.
- `AvgL1Norm` divides each vector by its mean absolute value with an epsilon clamp.
- `Actor.forward(state, zs)` first maps the raw state to a normalized feature, concatenates it with `zs`, and outputs a `tanh` normalized action.
- `Encoder.zs(state)` returns the normalized state embedding.
- `Encoder.zsa(zs, action)` returns the state-action embedding without applying AvgL1Norm to `zsa`.
- `Critic.forward(state, action, zsa, zs)` builds two Q networks. Each network receives normalized raw state-action features plus `[zsa, zs]`.
- `Agent.train()` updates the encoder, critic, LAP priority, delayed actor, and then performs hard target and fixed-encoder updates every `target_update_rate` steps.
- `fixed_encoder` is used for current actor and critic inputs. `fixed_encoder_target` is used for target values.
- `checkpoint_actor` and `checkpoint_encoder` are used at evaluation time when checkpointing is enabled.

The most important distinction from this repo's current TD3 is that TD7 is not a drop-in network width change. Its actor, critic, target update, replay priority, save/load payload, and evaluation policy state are all structurally different.

## Current Repo TD3 Implementation Map

The current continuous TD3 implementation is centered in `TD3Agent/agent.py`.

The runtime-facing TD3 API is already clean:

- `take_action(state, explore=False)`
- `act_eval(state, sigma_eval=0.0)`
- `push(s, a, r, ns, done)`
- `train_step(bc_context=None)`
- `save(directory, prefix="td3", include_optim=False)`
- `load(path)`

The current TD3 agent already includes several process-control additions:

- Mixed prioritized/recent/uniform replay through `PERRecentReplayBuffer`.
- Optional one-step, n-step, and truncated lambda modes.
- Gaussian or parameter-noise exploration.
- Huber or MSE critic loss.
- Soft or hard target updates.
- `bc_context` support for behavior cloning or handoff regularization.
- Diagnostics for action saturation, exploration, Q traces, losses, n-step returns, and BC loss.

The current actor is a standard MLP:

$$ a = \pi_{\theta}(s). $$

The current critic is a twin Q network:

$$ Q_{\phi_i} = Q_{\phi_i}(s,a). $$

The current TD3 target is:

$$ y = r + \gamma (1-d)\min_i Q_{\bar{\phi}_i}(s', \pi_{\bar{\theta}}(s')+\epsilon). $$

The current runners use the TD3 agent through a small number of shared patterns:

- Weights, residual, matrix, and structured matrix runners usually receive an already-built `runtime_ctx["agent"]`.
- These runners use `utils.agent_step_runtime.select_continuous_action` and `utils.agent_step_runtime.replay_train_continuous_agent`.
- Markov has `make_td3_markov_agent()` inside `utils/markov_runner.py`, because Markov correction currently has TD3-specific construction logic.
- Combined receives an `agents` dictionary and has per-family `agent_kind` flags, but the Markov branch currently enforces TD3 for the Markov supervisor.

This means the cleanest TD7 integration point is the agent API and config layer, not notebook-local training logic.

## Gap Analysis

The current TD3 implementation already has several components that can be reused conceptually, but TD7 should still be implemented as a new agent.

Reusable ideas:

- Runtime API shape.
- Current `bc_context` schema and diagnostics.
- Existing replay storage layout for `states`, `actions`, `rewards`, `next_states`, and `dones`.
- Current runner gating for warm start, hidden release, behavior cloning handoff, and training start.
- Current plotting and result-bundle convention for loss traces and agent metadata.

New TD7 pieces required:

- `AvgL1Norm`.
- `TD7Encoder` with `zs()` and `zsa()` methods.
- TD7 actor that accepts `(state, zs)`.
- TD7 critic that accepts `(state, action, zsa, zs)`.
- `encoder`, `fixed_encoder`, `fixed_encoder_target`, and `checkpoint_encoder`.
- `checkpoint_actor`.
- Encoder optimizer and encoder loss trace.
- Value clipping range variables `q_min`, `q_max`, `q_min_target`, and `q_max_target`.
- LAP-style priority update using the maximum twin TD error with a minimum priority.
- Hard target/fixed updates every `target_update_rate` steps.
- Save/load payloads that include actor, critic, encoder, fixed encoders, checkpoint networks, optimizers when requested, and TD7-specific range state.

Implementation risks:

- Directly loading TD3 checkpoints into TD7 is not valid because the actor and critic input shapes differ.
- Soft target updates are not the faithful TD7 default. The faithful first port should use hard copies at `target_update_rate`.
- The current `PERRecentReplayBuffer` uses importance weights and a mixed recent/uniform strategy. Fujimoto's LAP is simpler and uses priority without importance weights in the official code. The first TD7 port should either add a LAP-compatible mode to the existing buffer or create a `TD7ReplayBuffer` wrapper with the same export conventions.
- The current `multistep_mode` options are useful, but the first faithful TD7 validation should use one-step targets. Multi-step TD7 can be a later variant.
- Markov correction stores normalized raw action or executed raw action depending on config. TD7 should keep the rule that replay stores the action actually applied to the plant whenever safety projection or fallback changes the actor request.

## Proposed TD7 Architecture For This Repo

Add a new package adjacent to `TD3Agent`, named `TD7Agent`.

Minimum proposed files:

- `TD7Agent/encoder.py`
- `TD7Agent/actor.py`
- `TD7Agent/critic.py`
- `TD7Agent/agent.py`
- `TD7Agent/replay_buffer.py` only if the current replay buffer cannot support faithful LAP cleanly
- `TD7Agent/__init__.py`

The `TD7Agent` runtime API must match the TD3 API:

```python
agent.take_action(state, explore=False)
agent.act_eval(state, sigma_eval=0.0)
agent.push(s, a, r, ns, done)
agent.train_step(bc_context=None)
agent.save(directory, prefix="td7", include_optim=False)
agent.load(path)
```

The first implementation should use these default semantics:

- Actions remain normalized in `[-1, 1]`, matching the current TD3 runners.
- `take_action(..., explore=True)` uses Gaussian exploration first. Parameter-noise exploration can be added later if needed.
- `act_eval()` uses the checkpoint policy only when `use_checkpoint=True` or when a config flag says evaluation should use checkpoints. During early development, log both current-policy and checkpoint-policy returns when possible.
- Training uses one-step TD7 targets first.
- Actor updates use both critics averaged, matching the official TD7 policy loss.
- The BC penalty uses the current `bc_context` shape and coordinate weights.
- Diagnostics include `encoder_loss_trace`, `q_min_target_trace`, `q_max_target_trace`, `checkpoint_return_trace`, and `checkpoint_active_trace`.

The future config should add:

```python
"agent_kind": "td7"
"td7_agent": {
    "zs_dim": 256,
    "encoder_hidden": [256, 256],
    "actor_hidden": [256, 256],
    "critic_hidden": [256, 256],
    "target_update_rate": 250,
    "policy_delay": 2,
    "target_policy_smoothing_noise_std": 0.2,
    "noise_clip": 0.5,
    "replay_alpha": 0.4,
    "min_priority": 1.0,
    "bc_lambda_scale": 1.0,
    "use_checkpoints": true
}
```

These names can be adjusted during implementation, but the behavior should stay decision-complete: `td3_agent` remains for TD3 and `td7_agent` is the new TD7 config surface.

## Runner-By-Runner Integration Plan

### Residual Runner

Residual should be the first validation target after TD7 exists. The current May 31 distillation analysis shows residual is the strongest current TD3 runner because it improves both outputs and reward relative to OF-MPC in the tail.

Implementation plan:

- Allow `residual_cfg["agent_kind"]` to accept `"td7"` in addition to `"td3"` and `"sac"`.
- Build `TD7Agent` from `td7_agent` config when selected.
- Reuse the existing residual state, normalized residual action, reward, BC handoff, safety clipping, and replay storage rule.
- Compare against the existing TD3 residual run with the same seed, reward, warm start, and disturbance profile.

Key risk:

- Residual has an early-release crash mode. TD7 should be evaluated with the same phase-1 and BC handoff protections before any conclusion is drawn.

### Weights Runner

Weights is a natural second target because it uses the same continuous action runtime and has a clear TD3 action space.

Implementation plan:

- Allow `weight_cfg["agent_kind"] = "td7"`.
- Keep weight action bounds, multiplier cap, identity fallback, BC handoff, and shadow identity-MPC diagnostics unchanged.
- Reuse `bc_context` so TD7 can imitate the nominal or identity action during early training.

Key risk:

- TD7 may improve reward while still sacrificing composition. The evaluation must report output-specific MAE and outside-band fraction, not only scalar reward.

### Matrix And Structured Matrix Runners

Matrix and structured matrix are higher-risk than residual and weights because their actions modify model or prediction structure.

Implementation plan:

- Allow `"td7"` in the continuous agent-kind checks.
- Keep the current action mapping, caps, solve-failure fallback, and phase-1 hidden release behavior.
- Do not change matrix basis definitions during the TD7 integration.

Key risk:

- SALE learns a transition embedding over the RL state and chosen matrix action. If the matrix action is projected or replaced before execution, the replay buffer must store the executed action to avoid inconsistent transitions.

### Markov Correction Runner

Markov correction needs a special integration pass because `utils/markov_runner.py` currently has `make_td3_markov_agent()` and explicitly expects TD3 in the live proposal path.

Implementation plan:

- Rename the construction seam conceptually to `make_markov_continuous_agent()` or add a sibling `make_td7_markov_agent()`.
- Allow `agent_kind = "td7"` for the Markov runner.
- Keep the Markov action `z`, `z_bound`, basis family, LS fallback, z-safety projection, shadow safety diagnostics, and executed-action replay rule unchanged.
- Preserve the no-safeguard benchmark as TD3-only unless a new explicit TD7 benchmark file is added.

Key risk:

- The current Markov failure mode is a nearly fixed saturated direction. TD7 may learn a better representation, but it can also exploit the larger critic input in unsafe directions. Markov must keep safety and shadow diagnostics in the first TD7 experiments.

### Combined Supervisor

Combined should be integrated after the single-family TD7 agents are tested.

Implementation plan:

- Let `matrix_agent_kind`, `weights_agent_kind`, and `residual_agent_kind` accept `"td7"`.
- Let `markov_agent_kind` accept `"td7"` only after the single Markov runner is validated.
- Keep horizon agents as DQN or dueling DQN only.
- Keep SAC as a separate path and do not mix SAC changes into the TD7 pass.

Key risk:

- Multi-agent combined runs make attribution hard. Do not use combined as the first TD7 proof. Validate residual, weights, and Markov separately before enabling mixed TD3/TD7 combined runs.

## Test And Experiment Plan

Report validation:

- Confirm every local path listed in this report exists.
- Preview this file in VS Code Markdown Preview Enhanced.
- Check that equations render as display math.
- Check that relative image paths display if images are embedded in a later version.

Unit tests for the future TD7 implementation:

- Actor shape: `(batch, state_dim)` plus `(batch, zs_dim)` returns `(batch, action_dim)`.
- Critic shape: `(batch, state_dim)`, `(batch, action_dim)`, `(batch, zs_dim)`, and `(batch, zs_dim)` return two `(batch, 1)` tensors.
- Encoder shape: `zs(state)` and `zsa(zs, action)` both return `(batch, zs_dim)`.
- AvgL1Norm produces finite outputs for zeros, small values, and normal batches.
- Encoder loss backpropagates only through encoder parameters.
- Critic and actor losses do not update encoder parameters except through the explicit encoder optimizer step.
- Value clipping updates `q_min`, `q_max`, `q_min_target`, and `q_max_target` at the correct times.
- Priority update uses the maximum twin TD error and respects the minimum priority.
- `bc_context` changes the actor loss and records BC diagnostics.
- Save/load restores actor, critic, encoder, fixed encoders, checkpoint networks, optimizer state when requested, and value range state.

Smoke validation:

- First smoke-test TD7 on a small polymer run, following the repository rule to avoid opening the distillation Aspen column for smoke tests.
- Use one-step TD7 targets for the first smoke test.
- Verify no NaNs in actions, encoder loss, critic loss, actor loss, Q range, or priorities.
- Verify the replay buffer stores executed actions when a safety layer modifies the requested action.

Distillation validation after explicit user approval:

- Run residual TD3 and residual TD7 with identical run mode, disturbance profile, reward, seed, warm start, and test cycle.
- Compare reward, IAE, RMSE, tail MAE, outside-band fraction, input movement, saturation, encoder loss, Q range, TD error, and checkpoint-vs-current-policy behavior.
- Repeat only after residual is understood: weights, Markov, matrix or structured matrix, then combined.

Success criteria for TD7:

- TD7 must beat or match TD3 on tail tracking metrics, not only scalar reward.
- TD7 must not increase input movement enough to make the improvement unattractive for process control.
- TD7 must not rely on safety fallback to look stable.
- TD7 must provide useful diagnostics showing whether SALE embeddings are learning meaningful transition structure.

## Risks And Open Decisions

Risks:

- The official TD7 code is online/offline Gym-oriented. This repo has closed-loop MPC supervisors, warm starts, safety projections, BC handoffs, and disturbance scenarios. Faithfulness to TD7 must be balanced against preserving these process-control interfaces.
- TD7 value clipping assumes a meaningful tracked Q range. Reward scaling in this repo can be large, especially for distillation temperature reward. Reward scale and Q-range diagnostics are mandatory.
- The existing TD3 replay buffer already mixes priority, recent samples, and uniform samples. A fully faithful LAP port may require a TD7-specific buffer or a clearly documented LAP mode.
- Checkpointing in the TD7 paper evaluates policies across episodes. This repo has subepisodes, warm-start windows, and train/test schedules. The first checkpoint implementation should log enough information to verify whether checkpoint decisions are meaningful.
- Existing figures in this paper folder are conceptual. They should not be cited as empirical proof.

Open decisions for the implementation pass:

- Whether TD7 replay should reuse `PERRecentReplayBuffer` with a LAP mode or add a new `TD7ReplayBuffer`.
- Whether TD7 evaluation should use checkpoint policy by default for all test windows or only after a minimum number of training steps.
- Whether the checkpoint criterion should use subepisode reward, full episode reward, or a process-control metric such as tail band-normalized tracking cost.
- Whether near-setpoint replay sampling should be added immediately or deferred until baseline faithful TD7 is measured.
- Whether SALE sensitivity gates should be implemented in the first TD7 code pass or left as a second research variant.

## Recommended First Implementation Sequence

1. Add `TD7Agent` with faithful SALE, fixed encoders, target copies, value clipping, LAP priority, BC actor penalty, save/load, and the same runtime API as `TD3Agent`.
2. Add unit tests for shapes, loss routing, value clipping, priority update, and save/load.
3. Add `agent_kind = "td7"` and `td7_agent` config only for residual first.
4. Run a small polymer smoke test.
5. Run residual distillation only after explicit approval, then compare TD3 and TD7 under identical settings.
6. Add weights support after residual.
7. Add Markov support with safety diagnostics preserved.
8. Add matrix and structured matrix support.
9. Enable mixed TD3/TD7 combined supervisor variants only after single-runner behavior is understood.

## Bottom Line

TD7 is promising for this project because SALE adds a direct transition-structure learning signal to each process transition. That is aligned with nonlinear, delayed, constrained process control. However, TD7 should not be sold as an automatic offset-free solution. It should be implemented as a faithful parallel agent, evaluated first on residual RL-assisted MPC, and judged by tracking, input movement, safety intervention, and representation diagnostics rather than scalar reward alone.
