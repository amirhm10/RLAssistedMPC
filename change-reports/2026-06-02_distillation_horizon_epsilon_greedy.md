# Distillation Horizon Epsilon-Greedy Exploration

Date: 2026-06-02

## Summary

Changed the active distillation horizon DDQN defaults from NoisyNet exploration to epsilon-greedy exploration for the next standard and dueling horizon reruns.

## Changes

- `systems/distillation/notebook_params.py`
  - Standard horizon `exploration_mode` changed from `noisy` to `epsilon`.
  - Dueling horizon `exploration_mode` changed from `noisy` to `epsilon`.
  - Both horizon agents now use `eps_start = 0.20`, `eps_end = 0.02`, `eps_decay_mode = "linear"`, and `eps_decay_steps = 50000`.
- `distillation_RL_assisted_MPC_horizons_unified.py`
  - Standard DDQN now reads and passes `eps_decay_steps` into `DQNAgent`.
- `report/distillation_horizon_dqn_dueling_diagnosis_2026_06_02.md`
  - Added an exploration and horizon-range update section.

## Rationale

The latest horizon runs showed high late recipe churn under NoisyNet exploration. Epsilon-greedy makes the exploration rate explicit and easier to compare against saved action-switch diagnostics.

The horizon recipe range was not increased. Saved 263-recipe runs were poor, while the best current-reward 87-recipe runs concentrated around `(6, 3)` with some useful medium or long anchors such as `(11, 11)` and `(12, 7)`. The next run should keep the current 87-recipe grid to isolate the exploration change. If churn persists, a medium reduced grid such as `Np 4-12, Nc 2-11` is a better next ablation than a wider grid.

## Validation

- Compile-check the touched Python files.
- Import-check the active distillation notebook defaults and confirm both horizon agents resolve to epsilon-greedy with `0.20 -> 0.02` linear decay.

No Aspen simulation was launched.
