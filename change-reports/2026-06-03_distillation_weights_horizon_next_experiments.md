# Distillation Weights And Horizon Next-Experiment Report

Date: 2026-06-03

## Summary

Added a standalone analysis report for the latest distillation reruns:

- SG-TD3 weights with Gaussian exploration.
- Standard DDQN horizon with epsilon-greedy exploration.
- Dueling DDQN horizon with epsilon-greedy exploration.

The report concludes that the next high-value experiment is SG-DQN for the horizon family. It recommends dueling SG-DQN first, a paired standard SG-DQN ablation second, and no immediate horizon-range pruning until the gate logs show which recipes are actually trusted.

## Key Findings

- Latest SG-TD3 weights tail reward is `18.41`, up from `14.37` in the previous weights SG-TD3 run.
- Latest SG-TD3 weights has zero negative post-warm episodes and tail policy selection of `57.9%`.
- Latest standard DDQN horizon tail reward is `8.65`, but it has `13` negative post-warm episodes and a worst post-warm reward of `-22.86`.
- Latest dueling DDQN horizon tail reward is `9.29`, but it has `30` negative post-warm episodes and a worst post-warm reward of `-9.42`.
- Both saved horizon epsilon traces end near `0.133`, not `0.02`, because `eps_decay_steps = 50000` is applied in DQN action-call scale rather than plant-step scale.

## Files Added

- `report/scripts/analyze_distillation_latest_weights_horizon_20260603.py`
- `report/distillation_weights_horizon_next_experiments_2026_06_03.md`

## Local Generated Artifacts

The analysis script writes ignored report artifacts under:

- `report/figures/distillation_latest_weights_horizon_20260603/`

Main outputs:

- `reward_summary.csv`
- `tracking_summary.csv`
- `horizon_summary.csv`
- `horizon_top_pairs_tail.csv`
- `weights_sg_summary.csv`
- `manifest.json`
- Eight PNG figures for reward, tracking, horizon stability, epsilon, and SG gate diagnostics.

## Validation

Commands run:

```powershell
C:\Users\hamediaa\.conda\envs\rl-env\python.exe report\scripts\analyze_distillation_latest_weights_horizon_20260603.py
C:\Users\hamediaa\.conda\envs\rl-env\python.exe -m py_compile report\scripts\analyze_distillation_latest_weights_horizon_20260603.py
```

Also checked that the report and script do not contain semicolons, following the local report-writing rule.

No Aspen run was launched. Raw result bundles were not modified.
