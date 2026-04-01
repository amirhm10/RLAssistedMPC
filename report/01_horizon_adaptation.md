# Horizon Adaptation

## Scope

This report explains the DQN-based method that changes MPC prediction and
control horizons instead of directly changing continuous plant inputs.

## Notebooks Included

- `RL_assisted_MPC_horizons.ipynb`
- `RL_assisted_MPC_horizons_dist.ipynb`

## Inputs And Outputs

- Input state: scaled observer state, scaled setpoint deviation, scaled current
  input deviation
- Action: discrete horizon-recipe index
- Output behavior: rebuilt MPC with a new `(Hp, Hc)` pair

## Code Paths Explained

- `utils/helpers.py`
  `build_horizon_recipes`, `action_to_horizons`, `apply_rl_scaled`
- `DQN/dqn_agent.py`
  epsilon-greedy action selection and Double-DQN training step
- notebook-local `run_dqn_mpc_horizon_supervisor`

## Same-Page Caveats

- The main control loop is notebook-local, not module-first.
- The notebook family applies an extra `0.01` reward scaling.
- Disturbance variants reuse the same DQN scaffold but add scheduled process
  drift through `Qi`, `Qs`, and `hA`.
