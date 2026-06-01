# Distillation OF-MPC Current-Reward Figure

## Summary

Generated a standalone figure that rescales the saved disturbance-fluctuation OF-MPC trajectory with the current distillation reward defaults.

## Source Data

- `Distillation/Data/mpc_results_disturb_fluctuation.pickle`

## Reward Parameters

- `Q_diag = [37000, 20000]`
- `R_diag = [2500, 2500]`
- `k_rel = [0.3, 0.01]`
- `band_floor_phys = [0.003, 0.2]`
- `beta = 7.0`
- `reward_scale = 1.0`

## Outputs

- `report/figures/distillation_ofmpc_current_reward_20260601/fig_ofmpc_avg_reward_current_reward.png`
- `report/figures/distillation_ofmpc_current_reward_20260601/ofmpc_current_reward_avg.csv`
- `report/figures/distillation_ofmpc_current_reward_20260601/summary.json`
- `report/scripts/generate_distillation_ofmpc_current_reward_20260601.py`

## Result

- Mean average reward: `7.7192`
- Tail-20 average reward: `6.3910`
- Final average reward: `6.9262`

## Validation

- Regenerated the figure from the saved OF-MPC bundle only.
- Visually inspected the generated PNG.
- No Aspen/distillation runtime was launched.
