# Exploration Defaults: Gaussian TD3 and Epsilon DQN

Updated polymer and distillation exploration defaults so new runs prefer the
settings that have been more successful in the recent distillation experiments.

## Changes

- Continuous TD3-style runners now default to Gaussian action noise rather than
  parameter noise.
- Discrete DQN and dueling-DQN horizon runners now default to epsilon-greedy
  exploration rather than NoisyNet.
- `DuelingDQNAgent` now also defaults to epsilon-greedy when constructed without
  an explicit `exploration_mode`.
- The Markov TD3 factory fallback now assumes Gaussian exploration if an older
  config omits `exploration_mode`.

SAC and SG-SAC are unchanged: they still use SAC's stochastic policy sampling
during training and deterministic mean actions during evaluation.

## Validation

- `py_compile` for the edited defaults, Markov factory, DuelingDQN agent, and
  the new test.
- `tests/test_exploration_defaults.py`
- `tests/test_supervisor_gated_horizon_runners.py`
- `tests/test_supervisor_gated_dqn.py`

No polymer or distillation closed-loop runner was executed.
