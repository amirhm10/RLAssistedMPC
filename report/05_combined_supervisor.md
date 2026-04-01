# Combined Supervisor

## Scope

This report explains the multi-agent notebook family where DQN selects horizons
and two TD3 agents adapt the model and weights around one MPC solve.

## Notebooks Included

- `RL_assisted_MPC_combined.ipynb`
- `RL_assisted_MPC_combined_dist.ipynb`

## Inputs And Outputs

- Shared state: scaled observer state, scaled setpoint deviation, scaled input
  deviation
- Horizon action: discrete recipe index
- Model action: continuous `[alpha, delta]`
- Weight action: continuous `[Q1, Q2, R1, R2]` multipliers

## Code Paths Explained

- notebook-local `run_multi_agent_rl_mpc`
- `utils/helpers.py`
  `action_to_horizons`, `apply_rl_scaled`
- `TD3Agent/agent.py` and `DQN/dqn_agent.py`

## Same-Page Caveats

- The observer-update hook exists but is commented out in the combined notebook.
- The warm-start baseline behavior is part of the method.
- All three agents train from the same scalar reward.
