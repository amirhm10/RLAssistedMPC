# Distillation SG-DQN Horizon Follow-Up Report

Date: 2026-06-04

## Summary

Extended `report/distillation_dueling_horizon_history_2026_06_04.md` with the completed June 4 distillation SG-DQN horizon runs.

Key findings:

- Standard SG-DQN is now the strongest saved distillation horizon result in this follow-up, with tail reward `11.06` versus OF-MPC `6.39` and only one negative post-warm episode.
- SG-dueling-DQN is disappointing: tail reward is `6.35`, essentially OF-MPC level, with `43` negative post-warm episodes and worse T85 tracking than the previous dueling epsilon run.
- The corrected epsilon schedule worked; both SG runs reach tail epsilon `0.02`.
- The SG-dueling failure is therefore a Q-ranking and horizon-schedule issue, not an epsilon or old safety-layer issue.

## Files Changed

- `report/distillation_dueling_horizon_history_2026_06_04.md`
- `report/scripts/analyze_distillation_sg_dqn_horizon_followup_20260604.py`
- `report/figures/distillation_dueling_horizon_history_20260604/sg_dqn_followup_summary.csv`
- `report/figures/distillation_dueling_horizon_history_20260604/sg_dqn_followup_comparisons.csv`
- `report/figures/distillation_dueling_horizon_history_20260604/sg_dqn_followup_top_pairs.csv`
- `report/figures/distillation_dueling_horizon_history_20260604/sg_dqn_followup_summary.json`
- `report/figures/distillation_dueling_horizon_history_20260604/fig_sg_dqn_followup_*.png`

## Validation

```powershell
C:\Users\hamediaa\.conda\envs\rl-env\python.exe -m py_compile report\scripts\analyze_distillation_sg_dqn_horizon_followup_20260604.py
C:\Users\hamediaa\.conda\envs\rl-env\python.exe report\scripts\analyze_distillation_sg_dqn_horizon_followup_20260604.py
```
