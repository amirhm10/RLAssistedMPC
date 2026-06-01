# Supervisor-Gated TD3

## Objective

This implementation enables testing a supervisor-gated TD3 variant for RL-assisted MPC without changing the existing TD3 baselines. The method is designed for any continuous supervisor that can provide a full candidate action in the same normalized action space as the TD3 actor. The supervisor may be MPC, offset-free MPC, PID, a nominal steady-state action, a previous action, or another safe fallback.

Performance improvement is not claimed here. The implementation only adds the machinery needed for controlled experiments.

## Difference From Standard TD3

Standard TD3 executes the actor action during live policy control. Supervisor-gated TD3 forms two candidate actions:

$$ a_{\mathrm{rl},k} = \mu_{\theta}(s_k), \qquad a_{\mathrm{sup},k} = \mu_{\mathrm{sup}}(s_k). $$

Both actions are normalized TD3 action vectors:

$$ a_{\mathrm{rl},k}, a_{\mathrm{sup},k} \in [-a_{\max}, a_{\max}]^{n_u}. $$

The new method evaluates both candidates with the same two TD3 critics and executes the candidate with the better conservative score, subject to a configurable margin.

| Component | Existing TD3 | Supervisor-gated TD3 |
|---|---|---|
| Actor action | actor only | actor candidate |
| Supervisor action | not used | second candidate |
| Critics | two critics | same two critics |
| Execution | actor action | best conservative candidate |
| Actor loss | maximize Q | maximize Q plus supervisor-aware regularization |
| Critic loss | executed action | executed action only |

## Mathematical Formulation

For each candidate action, the critics provide:

$$ Q_1(s_k,a), \qquad Q_2(s_k,a). $$

The conservative value and critic disagreement are:

$$ Q_{\min}(s_k,a) = \min(Q_1(s_k,a), Q_2(s_k,a)), \qquad D_Q(s_k,a) = |Q_1(s_k,a) - Q_2(s_k,a)|. $$

The score is:

$$ S(s_k,a)=Q_{\min}(s_k,a)-\rho_Q D_Q(s_k,a)-\kappa_{\mathrm{prev}}\|a-a_{\mathrm{prev},k}\|_2^2-\kappa_{\mathrm{sup}}\|a-a_{\mathrm{sup},k}\|_2^2. $$

The actor is selected only if it proves an advantage:

$$ a_k = a_{\mathrm{rl},k} \quad \mathrm{if} \quad S(s_k,a_{\mathrm{rl},k}) > S(s_k,a_{\mathrm{sup},k}) + \epsilon_A. $$

Otherwise the supervisor action is executed. This makes the supervisor the default near steady state when many actions have nearly equal value.

The actor loss in the new agent is:

$$ \mathcal{L}_{\pi}=-\mathbb{E}[Q_{\mathrm{mode}}(s,\mu_{\theta}(s))]+\lambda_{\mathrm{sup}}\mathbb{E}[w_{\mathrm{sup}}\|\mu_{\theta}(s)-a_{\mathrm{sup}}\|_2^2]+\lambda_{\Delta a}\mathbb{E}[\|\mu_{\theta}(s)-a_{\mathrm{prev}}\|_2^2]. $$

The supervisor weight is:

$$ w_{\mathrm{sup}}(s)=\sigma\left(\frac{\epsilon_A-A_{\mathrm{rl|sup}}(s)}{\tau_{\mathrm{sup}}}\right). $$

By default, this weight is detached, and gradients do not pass through the supervisor action.

The critic target remains the standard one-step TD3 target:

$$ y_k = r_k + \gamma(1-d_k)\min_i Q_i^-(s_{k+1},\mu_{\theta^-}(s_{k+1})+\epsilon). $$

The supervisor action is not used in this target in version 1.

## Files Added

- `utils/supervisor_gated_action.py` contains pure helper functions for action validation, score computation, and policy-versus-supervisor selection.
- `TD3Agent/supervisor_replay_buffer.py` adds `SupervisorPERRecentReplayBuffer`, which inherits the current mixed replay behavior and stores supervisor-gate metadata.
- `TD3Agent/supervisor_gated_agent.py` adds `SupervisorGatedTD3Agent`, `SupervisorGateConfig`, and `SupervisorGatedDecision`.
- `tests/test_supervisor_gated_td3.py` adds lightweight agent and replay tests that do not require notebook execution, Aspen, or nonlinear plant simulation.

## Replay Metadata

The new replay buffer stores the executed transition in the same base arrays as the current TD3 replay buffer. It also records:

- `policy_actions`
- `supervisor_actions`
- `previous_actions`
- `selected_sources`
- `score_policy`
- `score_supervisor`
- `advantage_policy_supervisor`

The inherited `sample(...)` format is unchanged. New code uses `sample_supervised(...)` to receive a dictionary with the base transition tensors and the supervisor metadata.

## Why The Critic Uses Only Executed Actions

The critic is trained on the action that actually produced the next state. If the gate selects the supervisor, the replay action is the supervisor action. If the gate selects the policy, the replay action is the policy action. The implementation does not train the critic on both actions for the same next state, because that would create a counterfactual transition that was not observed.

## Future Notebook Use

A future notebook or runner should create a normalized supervisor action, call `select_action_with_supervisor(...)`, execute `decision.action`, and then call `push_supervised(...)` with the executed action plus the policy, supervisor, previous-action, source, score, and advantage metadata. Warm-start periods should execute the supervisor and store `SOURCE_WARM_START` if transitions are retained.

Initial polymer experiments should compare existing TD3, supervisor-only control, and supervisor-gated TD3 with and without positive advantage margins and critic-disagreement penalties. Distillation should not be the first smoke path.

## Initial Validation Results

The implementation includes compile and unit-test coverage for imports, action shape, supervisor tie behavior, replay metadata, one-step training, and original TD3 signature compatibility. These tests validate software behavior only. They do not establish closed-loop process-control performance.

Local validation used the available shell Python because the documented `rl-env` interpreter path was not present on this machine:

- `python -m compileall -q TD3Agent utils tests` passed.
- `python -m pytest tests/test_supervisor_gated_td3.py -q` could not run because `pytest` is not installed in the available interpreter.
- `python tests/test_supervisor_gated_td3.py` passed.

## Limitations

- Version 1 supports only one-step TD3 replay.
- The supervisor is not used in the TD3 target.
- No ranking loss is included.
- No generic runner is included yet.
- No polymer or distillation performance claims are made before controlled experiments.

## What Was Done

- Added an opt-in supervisor-gated TD3 agent.
- Preserved the existing TD3 implementation and runner behavior.
- Added supervisor metadata replay support.
- Added conservative twin-critic action scoring.
- Added supervisor-aware actor regularization.
- Added lightweight tests and implementation notes.
