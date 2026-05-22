# Distillation Residual, Markov, And Reward Focus Report

Date: 2026-05-21

## Summary

Added a focused analysis report for the latest distillation runs, centered on three questions:

- why residual TD3 helps polymer but fails on the distillation column
- why the latest distillation Markov run has strong scalar reward but strange diagnostics
- what reward-parameter changes are worth testing next

## Outputs

- Report: `report/distillation_residual_markov_reward_focus_2026_05_21.md`
- Analysis script: `report/scripts/analyze_distillation_residual_markov_reward_focus_20260521.py`
- Figure/data folder: `report/figures/distillation_residual_markov_reward_focus_20260521/`

## Main Findings

- Distillation residual TD3 saturates its raw residual action at the action bounds in the tail, while the authority layer projects every tail step. Under the current recomputed distillation reward, residual is `-12.40` tail reward units below OF-MPC.
- Polymer residual is also authority-limited, but the same saved-run comparison still improves tail reward by `+1.28`, which indicates the failure is plant/method interaction rather than residual code being inactive.
- Distillation Markov is scalar-reward competitive but diagnostically odd: TD3 is the tail source, requested z is frequently projected, T85 band-normalized error remains above one, and the requested cost guard passes only a small fraction of tail samples.
- Re-scoring saved distillation trajectories with `Q_T = 5000` and `Q_T = 10000` does not rescue residual. It mostly penalizes Markov and residual trajectories more strongly, while TD3 weights remains the best saved trajectory across the tested reward variants.

## Validation

- Regenerated all figures and CSV/JSON summaries from saved result bundles.
- Visually inspected generated figures for readability.
- Did not run new Aspen simulations.
