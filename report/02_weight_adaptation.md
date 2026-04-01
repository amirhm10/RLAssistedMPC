# Weight Adaptation

## Scope

This report explains the TD3-based method that changes MPC output and input
penalties online through four multiplicative coefficients.

## Notebooks Included

- `RL_assisted_MPC_weights.ipynb`
- `RL_assisted_MPC_weights_dist.ipynb`

## Inputs And Outputs

- Input state: scaled observer state, scaled setpoint deviation, scaled input
  deviation
- Action: four continuous coefficients for `[Q1, Q2, R1, R2]`
- Output behavior: same MPC structure, different stage-cost priorities

## Code Paths Explained

- notebook-local `run_rl_train_disturbance_gradually`
- `Simulation/mpc.py`
  `MpcSolverGeneral.mpc_opt_fun`
- `TD3Agent/agent.py`
  continuous action sampling and TD3 training

## Same-Page Caveats

- The notebook initializes the pre-warm-start action at unity multipliers.
- This family scales the reward by `0.01`.
- The structural model and horizon remain fixed; only the cost terms change.
