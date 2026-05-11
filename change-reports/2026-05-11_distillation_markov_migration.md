# 2026-05-11 Distillation Markov Migration

- added a new distillation Markov notebook family with TD3-only v1 defaults
- mapped the new distillation `markov` family to `C2S_SS_simulation11.dynf`
- created `distillation_RL_assisted_MPC_markov_unified.ipynb` using the shared Markov runner and shared Markov plotting path
- generalized the shared Markov runtime context builder so caller-supplied disturbance schedules are the primary path and polymer disturbance synthesis stays as a fallback
- added shared-system teardown support so distillation Markov runs can close Aspen sessions created through `system_factory`
- configured distillation Markov behavioral cloning as a short post-warm-start LS-targeted window
