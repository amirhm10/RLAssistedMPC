# Distillation Reward Audit

## Main finding

The archived distillation notebooks do not all use the same reward family.

- `RL_assisted_MPC_horizons*.ipynb`
- `RL_assisted_MPC_matrices*.ipynb`
- `RL_assisted_MPC_weights*.ipynb`

These archived RL notebooks use an inline `make_reward_fn_relative_QR(...)` and return:

```python
return -(err_eff + move + lin_out + lin_in) + bonus
```

That reward is unscaled in the archived notebooks.

## Relative-band RL family

For the archived distillation horizon, matrices, and weights notebooks, the effective reward parameters are consistent:

- `k_rel = [0.3, 0.02]`
- `band_floor_phys = [0.003, 0.3]`
- `Q_diag = [3.7e4, 1.5e3]`
- `R_diag = [2.5e3, 2.5e3]`
- `tau_frac = 0.7`
- `gamma_out = 0.5`
- `gamma_in = 0.5`
- `beta = 7.0`
- `gate = "geom"`
- `lam_in = 1.0`
- `bonus_kind = "exp"`
- `bonus_k = 12.0`
- `bonus_p = 0.6`
- `bonus_c = 20.0`
- archived effective scale: `1.0`

The canonical distillation data files match the archived files for this family:

- `system_dict.pickle`
- `scaling_factor.pickle`
- `min_max_states.pickle`
- nominal baseline pickle

So the main reward-range difference for these notebooks was the shared helper's fixed `* 0.01`, not a different model or scaler.

## Archived combined notebook

`DIstillation Column Case/RL_assisted_MPC_DL/RL_assisted_MPC_combined.ipynb` does **not** use the relative-band RL reward above. It uses `make_reward_fn_fixed_QR(...)`.

Effective archived combined reward setup:

- `Q_diag = [60000.0, 30000.0]`
- `R_diag = [1000.0, 1000.0]`
- `y_band_phys = [0.0001, 0.1]`
- `tau_frac = 0.7`
- `gamma_out = 0.7`
- `gamma_in = 0.7`
- `beta = 50.0`
- `gate = "geom"`
- `lam_in = 1.0`
- `bonus_kind = "exp"`
- `bonus_k = 3.0`
- archived effective scale: `1.0`

This means the archived combined distillation reward range will not match the unified combined notebook unless the unified combined path is also switched to a fixed-band reward profile.

## Archived baseline MPC notebooks

- `MPCOffsetFree.ipynb`
- `MPCOffsetFreeDistRamp.ipynb`
- `MPCOffsetFreeDistRampNoise.ipynb`

These also use `make_reward_fn_fixed_QR(...)`, not the relative-band RL reward.

One archived nominal baseline setup is:

- `Q_diag = [100.0, 100.0]`
- `R_diag = [1.0, 1.0]`
- `y_band_phys = [0.002, 0.002]`
- `tau_frac = 0.2`
- `gamma_out = 0.7`
- `gamma_in = 0.7`
- `beta = 50.0`
- `gate = "prod"`
- `lam_in = 1.0`
- `bonus_kind = "exp"`
- `bonus_k = 4.0`
- archived effective scale: `1.0`

So the unified distillation baseline reward range will also differ from the archived baseline notebooks unless the unified baseline path is moved onto that fixed-band reward family.

## Practical consequence

The new `reward_scale` fix aligns the shared helper with the archived distillation RL notebook family that used `make_reward_fn_relative_QR(...)`.

It does **not** by itself align the unified distillation:

- baseline MPC notebook
- combined notebook

with their archived counterparts, because those archived notebooks use a different reward family and different parameter sets.
