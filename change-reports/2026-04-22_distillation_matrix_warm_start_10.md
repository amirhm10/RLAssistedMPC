# 2026-04-22 Distillation Matrix Warm Start 10

- Updated [systems/distillation/config.py](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/systems/distillation/config.py) so the shared `DISTILLATION_MATRIX_RUN_PROFILES` use `warm_start = 10` instead of `5`.
- This applies to both distillation matrix families that read the shared matrix run profiles:
  - [distillation_RL_assisted_MPC_matrices_unified.ipynb](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/distillation_RL_assisted_MPC_matrices_unified.ipynb)
  - [distillation_RL_assisted_MPC_structured_matrices_unified.ipynb](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/distillation_RL_assisted_MPC_structured_matrices_unified.ipynb)
- Both TD3 and SAC nominal/disturbance profiles now keep the baseline MPC action for the first 10 sub-episodes before the RL matrix policy takes over.
