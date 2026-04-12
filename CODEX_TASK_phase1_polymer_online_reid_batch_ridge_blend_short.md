# Codex task for `RLAssistedMPC`: phase 1 polymer online batch re-identification with RL blend

Implement a **new additive workflow** in the **current `RLAssistedMPC` repo** for the **polymer case only**.

Do **not** use the old `RL_controller` layout. Do **not** build this on stale legacy loops. Follow the current unified surface described in:

- `AGENTS.md`
- `report/notebook_refactor_audit.md`
- `utils/matrix_runner.py`
- `utils/state_features.py`
- `utils/helpers.py`
- `utils/observer.py`
- `Simulation/mpc.py`

## Main idea

Create a new polymer-only method where:

1. recent closed-loop data is used to **re-identify a candidate physical model online** using **batch ridge regression**
2. optional bounded constrained least-squares may be supported behind a config flag
3. RL does **not** output plant inputs in this workflow
4. RL outputs a scalar **blend factor** `eta in [0, 1]`
5. `eta` blends the nominal physical model and the identified physical model for **MPC prediction only**
6. the **observer must remain nominal** at all times

## Most important rule

The observer must stay nominal.

Use only:

- nominal `A_aug0`, `B_aug0`, `C_aug`
- nominal `L_nom`

Do **not** use the identified model or blended model in the observer update.

This repo already has the correct pattern in `utils/matrix_runner.py`:

- assisted prediction model is changed
- fixed observer model is kept separate
- `recalculate_observer_on_matrix_change` is ignored

Follow that exact design principle.

## Architecture to follow

Use the unified-runner style already used in this repo.

Recommended additions:

1. `utils/reid_batch.py`
   - rolling ID buffer
   - polymer basis builder
   - regression assembly
   - batch ridge solver
   - optional bounded LS solver
   - model reconstruction
   - safety checks and fallback logic

2. `utils/reid_batch_runner.py`
   - main online re-identification + RL blend runtime
   - same style as `utils/matrix_runner.py`
   - same normalized result-bundle pattern

3. `RL_assisted_MPC_reid_batch_unified.ipynb`
   - new polymer-only unified notebook entrypoint

4. `report/05_online_reidentification_batch_ridge_blend.tex`
   - full math and implementation report

5. `change-reports/`
   - add a short markdown note because `AGENTS.md` asks for one for major changes

## Math to implement

Nominal physical model:

$$
 x_{k+1} = A_0 x_k + B_0 u_k
$$

Nominal offset-free augmentation:

$$
 A_{aug,0} = \begin{bmatrix} A_0 & 0 \\ 0 & I \end{bmatrix},
 \quad
 B_{aug,0} = \begin{bmatrix} B_0 \\ 0 \end{bmatrix},
 \quad
 C_{aug} = \begin{bmatrix} C & I \end{bmatrix}
$$

Nominal observer update only:

$$
 \hat{x}^{aug}_{k+1} = A_{aug,0}\hat{x}^{aug}_k + B_{aug,0}u_k^{dev} + L_{nom}(y_k^{dev} - C_{aug}\hat{x}^{aug}_k)
$$

Identify only the **physical** block of the state.

Use a linear-in-parameters family:

$$
 A(\theta) = A_0 + \sum_i \theta_i^A E_i,
 \quad
 B(\theta) = B_0 + \sum_j \theta_j^B F_j
$$

Use samples in scaled deviation coordinates:

$$
 (\hat{x}^{phys}_t, u_t^{dev}, \hat{x}^{phys}_{t+1})
$$

with residual

$$
 r_t = \hat{x}^{phys}_{t+1} - A_0\hat{x}^{phys}_t - B_0u_t^{dev}
$$

and regression form

$$
 r_t \approx \Phi_t \theta
$$

Batch ridge over a rolling window:

$$
 \hat{\theta}_k = \arg\min_{\theta}
 \sum_{t=k-W}^{k-1} \omega_t \|r_t - \Phi_t\theta\|_2^2
 + \lambda_{prev}\|\theta - \hat{\theta}_{k-1}\|_2^2
 + \lambda_0\|\theta\|_2^2
$$

Default solve mode:

- closed-form ridge solve
- then clip `theta` to bounds

Optional second mode:

- bounded constrained least squares with box bounds on `theta`

Build candidate model:

$$
 A_k^{id} = A(\hat{\theta}_k),
 \quad
 B_k^{id} = B(\hat{\theta}_k)
$$

RL blend:

- actor output `a_k in [-1, 1]`
- map to

$$
 \eta_k^{raw} = \frac{a_k + 1}{2}
$$

- smooth:

$$
 \eta_k = (1 - \tau_\eta)\eta_{k-1} + \tau_\eta \eta_k^{raw}
$$

Blended physical prediction model:

$$
 A_k^{pred} = A_0 + \eta_k(A_k^{id} - A_0)
$$

$$
 B_k^{pred} = B_0 + \eta_k(B_k^{id} - B_0)
$$

Then rebuild the **augmented prediction model for MPC only**.

## Phase-1 polymer basis

First inspect whether the current matrix workflow already has reusable polymer basis / mask logic.

If not, use this fallback basis:

For `A`, row scaling basis

$$
 E_i = e_i e_i^T A_0
$$

For `B`, input-column scaling basis

$$
 F_j = B_0 e_j e_j^T
$$

## Algorithm

At each step:

1. use nominal observer to estimate state
2. build RL state
3. actor outputs `eta`
4. build blended MPC prediction model
5. solve MPC using blended prediction model only
6. apply plant input
7. update nominal observer only
8. push `(x_phys_t, u_dev_t, x_phys_{t+1})` into rolling ID buffer
9. every `id_update_period` steps, solve batch ridge / bounded LS
10. update candidate identified model if valid
11. compute reward and store RL transition
12. train agent

## RL state

Reuse the current shared state-feature layer if possible.

Include at least:

- current nominal observer state
- current setpoint
- previous input
- previous `eta`
- identification residual norm
- `||Delta A||_F`
- `||Delta B||_F`
- fallback / invalid-solve flag

## Logging requirements

The result bundle must include at least:

- `eta_log`
- raw actor actions
- `theta_hat_log` or periodic theta history
- residual fit norm
- `A_model_delta_ratio_log`
- `B_model_delta_ratio_log`
- fallback flags
- invalid solve count
- rewards
- `delta_y_storage`
- `delta_u_storage`
- `xhatdhat`
- `yhat`
- `estimator_mode = "fixed_nominal"`
- `prediction_model_mode = "online_reid_blend"`

## Validation requirements

Must include these checks:

1. `eta = 0` for all steps reproduces nominal MPC to numerical tolerance
2. observer stays nominal even when prediction model changes
3. if ID is disabled or has insufficient data, workflow falls back to nominal MPC
4. synthetic theta recovery sanity check for the batch ridge solver
5. polymer smoke test runs successfully
6. comparison plots versus nominal MPC

## LaTeX report requirements

Create:

- `report/05_online_reidentification_batch_ridge_blend.tex`

The report must include:

1. motivation
2. repo-specific architecture discussion
3. explicit fixed-observer rule
4. full mathematical derivation
5. batch ridge derivation
6. optional constrained LS derivation
7. physical-only identification logic
8. RL blend logic
9. full algorithm
10. mapping to repo files
11. validation plan
12. limitations and next steps

## References to include

1. Ljung, *System Identification: Theory for the User*, 2nd ed., 1999
2. Hoerl and Kennard, "Ridge Regression: Biased Estimation for Nonorthogonal Problems," 1970
3. Rawlings, Mayne, and Diehl, *Model Predictive Control: Theory, Computation, and Design*, 2nd ed., 2022
4. Astrom and Wittenmark, *Adaptive Control*, 2nd ed., 1995
5. Sutton and Barto, *Reinforcement Learning: An Introduction*, 2nd ed., 2018

## Acceptance criteria

The task is complete only if:

1. existing unified workflows remain unchanged
2. the implementation is on the current unified runner surface, not stale legacy code
3. a new polymer-only unified workflow exists
4. RL controls `eta`, not plant inputs, in this workflow
5. identification uses only physical states in scaled deviation coordinates
6. the observer remains nominal
7. a detailed LaTeX report is created
8. a short change note is added under `change-reports/`
9. the `eta = 0` baseline matches nominal MPC