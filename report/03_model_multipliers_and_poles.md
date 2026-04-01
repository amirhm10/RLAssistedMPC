# Model Multipliers And Poles

## Scope

This report groups the methods that adapt the internal MPC prediction layer:
matrix multipliers with TD3 or SAC, model-mismatch variants, and observer-pole
adaptation.

## Notebooks Included

- `RL_assisted_MPC_matrices.ipynb`
- `RL_assisted_MPC_matrices_dist.ipynb`
- `RL_assisted_MPC_matrices_SAC.ipynb`
- `RL_assisted_MPC_matrices_dsit_SAC.ipynb`
- `RL_assisted_MPC_matrices_model_mismatch.ipynb`
- `RL_assisted_MPC_Poles.ipynb`

## Inputs And Outputs

- Matrix method action: `[alpha, delta_1, ..., delta_nu]`
- Pole method action: one pole proposal per observer state
- Output behavior: modified `A/B` model or modified observer gain `L`

## Code Paths Explained

- notebook-local `run_rl_train_disturbance_gradually`
- `Simulation/mpc.py`
  `compute_observer_gain`
- `TD3Agent/agent.py` and `SACAgent/sac_agent.py`

## Same-Page Caveats

- The matrix method and pole method share the observer-state pipeline but adapt
  different parts of the control stack.
- SAC changes the learning rule, not the overall supervisory loop.
- The mismatch notebook increases operating-condition drift to stress the model.
