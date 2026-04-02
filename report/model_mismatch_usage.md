# Model Mismatch Usage In The Legacy Mismatch Notebooks

This note explains what "mismatch" means in the old mismatch notebooks and how it is actually used in the code.

## Notebooks covered

- `RL_assisted_MPC_matrices_model_mismatch.ipynb`
- `RL_assisted_MPC_residual_model_mismatch.ipynb`
- `RL_assisted_MPC_residual_model_mismatch1.ipynb`
- `RL_assisted_MPC_residual_model_mismatch2.ipynb`
- `RL_assisted_MPC_residual_model_mismatch_multi.ipynb`

## What mismatch means here

In these notebooks, mismatch means the plant is moved away from the nominal operating condition while the controller still starts from the same identified linear MPC model.

The mismatch is injected on the plant side by changing process/design parameters during rollout:

- `Qi`
- `Qs`
- `hA`
- `CMf`

So the controller's linear model is no longer a perfect representation of the plant dynamics being simulated.

## 1. Matrix mismatch notebook

In `RL_assisted_MPC_matrices_model_mismatch.ipynb`, mismatch is used in two ways at once.

First, the plant is drifted away from nominal:

- `Qi`, `Qs`, and `CMf` are generated with `generate_disturbances_ramp(...)`
- `hA` is also ramped over the run
- during rollout the plant gets updated with:
  - `system.hA = ha[i]`
  - `system.Qs = qs[i]`
  - `system.Qi = qi[i]`
  - `system.CMf = cmf[i]`

The `CMf` mismatch in this notebook is not a simple straight-line ramp. It comes from `generate_disturbances_ramp(...)`, which builds piecewise smooth random-walk style ramps inside bounded relative ranges:

- `qi_rel=(0.9, 1.1)`
- `qs_rel=(0.9, 1.1)`
- `cmf_rel=(0.90, 1.10)`

Second, the RL agent is asked to compensate for that mismatch by changing the controller model itself:

- action dimension is `3`
- action is mapped to:
  - one scalar `alpha`
  - one per-input multiplier vector `delta`
- then the MPC model is changed each step:
  - `A_change[:n_phys, :n_phys] *= alpha`
  - `B_change[:n_phys, :] *= delta.reshape(1, -1)`

So the matrix mismatch notebook is doing:

- plant mismatch injection
- plus adaptive model correction inside MPC

### RL state in the matrix mismatch notebook

The mismatch notebook does not keep the standard unified state. It augments the RL state with extra mismatch signals:

- base state from `apply_rl_scaled(...)`
- `innov = y_prev_scaled - yhat_pred`
- `e_track = y_prev_scaled - y_sp[i, :]`

So the effective RL state is:

- estimated augmented state
- setpoint
- current input deviation
- prediction innovation
- tracking error

That is why `STATE_DIM = int(A_aug.shape[0]) + 3 * set_points_number + inputs_number`.

## 2. Residual mismatch notebooks

The residual mismatch family uses mismatch more aggressively than the matrix notebook.

These notebooks do not only inject mismatch into the plant. They also feed mismatch-derived signals into the residual RL policy and use them to control how much residual authority the policy is allowed to apply.

### Common residual structure

In the base residual mismatch notebooks, the MPC still computes a baseline action first:

- solve MPC
- get `u_base`

Then RL proposes a residual correction move:

- `delta_u_res_raw`

The final plant input becomes:

- `u_applied = u_base + delta_u_res`

So the mismatch idea here is:

- keep nominal MPC as the backbone
- let RL correct the mismatch with an additive residual move

### How mismatch enters the residual state

The residual mismatch notebooks augment the RL state with mismatch-aware features:

- `innov = y_prev_scaled - yhat_pred`
- `e_track = y_prev_scaled - y_sp[i, :]`

Then they concatenate those onto the usual scaled RL state:

- `current_rl_state = [standard_state, innov, e_track]`

So mismatch is not only something happening in the plant. It is explicitly exposed to the RL agent as part of the observation.

### How mismatch changes the residual action limits

The residual family also uses mismatch to decide how strong the residual action is allowed to be.

The notebooks define output bands with:

- `compute_band_scaled(...)`

Then they compute a mismatch/violation measure from tracking error:

- `rho = max(|e_track| / eps_i)`

That `rho` is used to scale the residual authority:

- `mag = (rho * beta_res) * (abs(delta_u_mpc) + du0_res)`

Then the raw residual proposal is clipped by:

- mismatch-scaled magnitude limits
- actuator headroom around `u_base`

So in the base residual mismatch design, mismatch is used in three places:

- plant parameter drift
- RL observation
- residual-action authority

### Executed residual is what gets stored

An important detail in these notebooks is that replay does not store the raw proposed residual action.

They compute the executed residual after clipping:

- `delta_u_res_exec = u_applied_scaled_abs - u_base`

Then they map that executed residual back into normalized action space and store it in replay:

- `a_exec = map_from_bounds(delta_u_res_exec, low_res, high_res)`

So the replay buffer reflects the residual move that was actually applied after safety clipping, not just the actor proposal.

## 3. Differences across the residual mismatch variants

### `RL_assisted_MPC_residual_model_mismatch.ipynb`

This is the main residual mismatch pattern:

- mismatch in plant via `Qi`, `Qs`, `hA`, `CMf`
- RL state includes `innov` and `e_track`
- residual authority depends on `rho`

### `RL_assisted_MPC_residual_model_mismatch1.ipynb`

This is a simpler residual-authority variant:

- still uses mismatch in plant
- still uses `innov` and `e_track` in the RL state
- but the residual magnitude bound is not multiplied by `rho`
- it uses:
  - `mag = beta_res * (abs(delta_u_mpc) + du0_res)`

So variant `1` still uses mismatch in the state, but less directly in the residual clipping rule.

### `RL_assisted_MPC_residual_model_mismatch2.ipynb`

This variant adds one more mismatch-aware signal:

- `de_track = e_track - e_prev`

That is why its state dimension is larger:

- `STATE_DIM = int(A_aug.shape[0]) + 4 * set_points_number + inputs_number`

So variant `2` uses:

- innovation
- tracking error
- change in tracking error

inside the RL observation.

### `RL_assisted_MPC_residual_model_mismatch_multi.ipynb`

This is not just another residual notebook. It is a combined mismatch notebook.

It uses two TD3 agents:

- one residual agent for additive correction
- one multiplier agent for `A/B` scaling

So this notebook combines:

- residual correction under mismatch
- matrix scaling under mismatch

inside one rollout.

Mismatch therefore drives:

- plant drift
- residual-state augmentation
- residual clipping logic
- matrix-scaling adaptation

all at once.

## 4. The role of `CMf`

`CMf` is the clearest "true model mismatch" variable in these notebooks because it changes a plant design parameter that the nominal controller model does not re-identify online.

How it is used:

- matrix mismatch notebook:
  - `CMf` is generated by `generate_disturbances_ramp(...)`
  - then applied with `system.CMf = cmf[i]`
- residual mismatch notebooks:
  - `CMf` is linearly ramped from `nominal_cmf` to `nominal_cmf * cmf_change`
  - then applied with `system.CMf = cmf[i]`

So `CMf` is part of the plant-side mismatch schedule in all of the residual mismatch notebooks, and also in the matrix mismatch notebook.

## 5. How this differs from the new unified notebooks

The new unified notebooks moved away from this mismatch-heavy observation design.

What we removed in the new unified paths:

- no `innov` appended to RL state
- no `e_track` appended to RL state
- no `de_track` appended to RL state
- no mismatch-band gating logic in the residual unified path
- no special mismatch reward path

What we kept in the new unified paths:

- the standard shared RL state:
  - augmented observer state
  - setpoint
  - input deviation
- the shared reward construction from `utils.rewards`
- disturbance scheduling through the standard `Qi/Qs/hA` path
- residual as an additive correction on top of MPC

So the old mismatch notebooks are more than just "disturbance notebooks". They are legacy experiments where mismatch is built into:

- the plant
- the RL observation
- and, in the residual family, the safety logic for residual execution

## Bottom line

If I compress the notebook intent into one sentence per family:

- `RL_assisted_MPC_matrices_model_mismatch.ipynb`:
  - mismatch is plant drift plus RL-driven rescaling of the MPC model matrices.
- `RL_assisted_MPC_residual_model_mismatch*.ipynb`:
  - mismatch is plant drift plus mismatch-aware residual correction, where the RL agent sees innovation/tracking-error terms and the residual move is clipped according to mismatch-related logic.
- `RL_assisted_MPC_residual_model_mismatch_multi.ipynb`:
  - mismatch is handled by both residual correction and matrix scaling at the same time.
