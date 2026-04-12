# Codex plan for `RLAssistedMPC`: phase 1 polymer online re-identification with batch ridge / constrained least squares and RL blend

## 1. Read this first

This repo is **not** the older `RL_controller` layout. Before implementing anything, inspect and follow the current unified surface described by:

- `AGENTS.md`
- `report/notebook_refactor_audit.md`
- `utils/matrix_runner.py`
- `utils/state_features.py`
- `utils/helpers.py`
- `utils/observer.py`
- `Simulation/mpc.py`

Important repo facts already confirmed from the current codebase:

1. The active implementation surface is the **unified notebook + shared utils runner** layer, not the old notebook-local split copies.
2. `Simulation/rl_sim.py` is considered stale / legacy and should **not** be the main extension target.
3. `utils/matrix_runner.py` already implements a crucial pattern we must preserve: it separates the assisted MPC prediction model from the fixed nominal observer model.
4. `utils/state_features.py` is already the shared home for RL state construction and should be reused or extended.
5. `utils/helpers.py` is already the shared home for scaling, system loading, and disturbance scheduling.
6. `utils/observer.py` is the current shared home for observer gain calculation.

This new method must therefore be implemented in the **same architectural style** as the unified matrix workflow.

## 2. Goal

Implement a **new additive workflow** for the **polymer case only** in phase 1.

The new method should do the following:

1. Use a rolling batch of recent closed-loop data to compute a **candidate re-identified physical model** online.
2. Use **batch ridge regression** as the default estimator.
3. Optionally support a **bounded constrained least-squares** variant behind a config flag.
4. Let RL control only a bounded **blend factor** `eta`, not the plant input directly.
5. Use the blend factor to combine the nominal physical model and the identified physical model for **MPC prediction only**.
6. Keep the **observer fully nominal** and isolated from the identified / blended model.
7. Keep all existing workflows, notebooks, and methods intact.

This is an **additive research extension**, not a rewrite of the existing matrix-multiplier workflow.

## 3. Non-negotiable repo-specific rules

### 3.1 Additive only

- Do not break the current unified notebooks.
- Do not change the behavior of horizon, matrix, weights, residual, or combined workflows.
- Prefer new modules and a new unified notebook entrypoint.
- If any shared file must be modified, make the change backward-compatible and opt-in.

### 3.2 Polymer only for phase 1

- This plan is only for the polymer system.
- Do not generalize to distillation in the first pass.
- Use explicit naming such as `polymer_phase1_*` in configs and saved artifacts.

### 3.3 Observer must stay nominal

This is the most important constraint.

The observer must use only the nominal augmented model and nominal observer gain.

Do **not** update the observer with:

- the re-identified model
- the blended prediction model
- any RL-controlled matrix

This repo already has the right pattern in `utils/matrix_runner.py`:

- assisted prediction model stored in `A_candidate`, `B_candidate`
- fixed observer model stored in `A_est`, `B_est`
- `L_nom` computed from the nominal model
- field `recalculate_observer_on_matrix_change` explicitly ignored

The new method must preserve this same logic.

### 3.4 Use the shared unified runner style

The new method should **not** be implemented as a notebook-only pile of code if it can be avoided.

Follow the current unified design:

- shared runner under `utils/`
- shared state builder under `utils/state_features.py` if needed
- new notebook entrypoint as a thin experiment launcher
- plotting and saved bundles aligned with the current result-bundle style

### 3.5 Use scaled deviation coordinates

Use the controller-side scaled deviation coordinates already used in the repo:

- state estimate in controller coordinates
- setpoints in scaled deviation coordinates
- applied input in scaled deviation coordinates

Do not re-identify in plant physical units unless there is a very strong reason.

## 4. Where this should live in the current repo

### 4.1 Main implementation targets

Add new modules under `utils/` rather than building on the stale legacy loop.

Recommended additions:

1. `utils/reid_batch.py`
   - rolling identification buffer
   - basis construction
   - feature assembly
   - batch ridge solver
   - optional bounded least-squares solver
   - model reconstruction
   - correction safety checks

2. `utils/reid_batch_runner.py`
   - main polymer online re-identification + RL blend runtime
   - same normalized result-bundle style as `utils/matrix_runner.py`
   - same train/test, warm-start, reward, and logging style as the unified runners

3. `utils/state_features.py`
   - extend only if needed for a new `state_mode` or a new helper dedicated to blend-policy state construction
   - keep backward compatibility

4. `RL_assisted_MPC_reid_batch_unified.ipynb`
   - a new unified notebook entrypoint for this method
   - polymer only for phase 1

5. `report/05_online_reidentification_batch_ridge_blend.tex`
   - full math and implementation report

6. `change-reports/`
   - because `AGENTS.md` asks for a short markdown note for major repo changes, add a brief implementation note there as well

### 4.2 Existing files to reuse, not replace

Reuse these where possible:

- `utils/helpers.py`
- `utils/observer.py`
- `utils/state_features.py`
- `Simulation/mpc.py`
- `TD3Agent/agent.py` or `SACAgent/sac_agent.py` depending on notebook choice
- the current result-bundle and plotting conventions used by unified runners

Do not create a parallel legacy framework unless absolutely necessary.

## 5. Mathematical formulation

## 5.1 Nominal physical and augmented model

Let the nominal physical model be

$$
 x_{k+1} = A_0 x_k + B_0 u_k
$$

with outputs

$$
 y_k = C x_k + d_k
$$

and the offset-free augmentation already used in the repo:

$$
 A_{aug,0} = \begin{bmatrix} A_0 & 0 \\ 0 & I \end{bmatrix},
 \quad
 B_{aug,0} = \begin{bmatrix} B_0 \\ 0 \end{bmatrix},
 \quad
 C_{aug} = \begin{bmatrix} C & I \end{bmatrix}.
$$

The observer update must remain nominal:

$$
 \hat{x}^{aug}_{k+1}
 = A_{aug,0} \hat{x}^{aug}_k + B_{aug,0} u_k^{dev}
 + L_{nom} \left(y_k^{dev} - C_{aug} \hat{x}^{aug}_k\right).
$$

This is not optional. This must be enforced in code structure.

## 5.2 Identify only the physical block

Let `n_y` be the number of outputs and `n_aug` the augmented dimension.
Then the physical dimension is

$$
 n_x = n_{aug} - n_y.
$$

Only the physical block is re-identified:

$$
 x_k^{phys} = \hat{x}^{aug}_k[0:n_x].
$$

Do **not** identify the disturbance / integrator block.

## 5.3 Linear-in-parameter re-identification family

The identified physical model must remain linear in the parameter vector:

$$
 A(\theta) = A_0 + \sum_{i=1}^{n_A} \theta_i^A E_i,
$$

$$
 B(\theta) = B_0 + \sum_{j=1}^{n_B} \theta_j^B F_j.
$$

Collect parameters as

$$
 \theta = \begin{bmatrix} \theta^A \\ \theta^B \end{bmatrix}.
$$

This is necessary so that the first re-identification method remains convex and easy to debug.

## 5.4 Basis selection for phase 1 polymer

Before implementing a new basis, inspect the current matrix workflow and reuse any existing polymer matrix mask logic if it is already cleanly reusable.

If that is not cleanly possible, use the following fallback basis for phase 1.

### A basis: row scaling

$$
 E_i = e_i e_i^T A_0,
 \quad i = 1, \dots, n_x
$$

so that

$$
 A(\theta^A) = A_0 + \sum_{i=1}^{n_x} \theta_i^A E_i.
$$

### B basis: input-column scaling

$$
 F_j = B_0 e_j e_j^T,
 \quad j = 1, \dots, n_u
$$

so that

$$
 B(\theta^B) = B_0 + \sum_{j=1}^{n_u} \theta_j^B F_j.
$$

This gives a simple, low-dimensional, linear-in-parameter model family suitable for phase 1.

## 5.5 Identification data in this repo

Use recent tuples from the nominal observer trajectory:

$$
 \left(\hat{x}^{phys}_t, u_t^{dev}, \hat{x}^{phys}_{t+1}\right).
$$

For each sample define the nominal one-step residual:

$$
 r_t = \hat{x}^{phys}_{t+1} - A_0 \hat{x}^{phys}_t - B_0 u_t^{dev}.
$$

For the selected basis, define the feature block:

$$
 \Phi_t = \begin{bmatrix}
 E_1 \hat{x}^{phys}_t & \cdots & E_{n_A} \hat{x}^{phys}_t &
 F_1 u_t^{dev} & \cdots & F_{n_B} u_t^{dev}
 \end{bmatrix}.
$$

Then

$$
 r_t \approx \Phi_t \theta.
$$

## 5.6 Batch ridge regression

For a rolling window of `W` samples, solve

$$
 \hat{\theta}_k = \arg\min_{\theta}
 \sum_{t=k-W}^{k-1} \omega_t \|r_t - \Phi_t \theta\|_2^2
 + \lambda_{prev} \|\theta - \hat{\theta}_{k-1}\|_2^2
 + \lambda_0 \|\theta\|_2^2.
$$

Recommended default:

- `omega_t = 1`
- regularize toward the previous estimate to suppress jitter

### Default solve mode

Implement default mode as:

1. closed-form ridge solve
2. then clip `theta` to configured bounds

### Optional second mode

Optional bounded constrained least-squares mode:

$$
 \min_{\theta}
 \sum_{t=k-W}^{k-1} \omega_t \|r_t - \Phi_t \theta\|_2^2
 + \lambda_{prev} \|\theta - \hat{\theta}_{k-1}\|_2^2
 + \lambda_0 \|\theta\|_2^2
$$

subject to

$$
 \theta_{min} \le \theta \le \theta_{max}.
$$

Implement this only if it is clean and stable. Keep the ridge-then-clip path as the default phase-1 method.

## 5.7 Candidate identified model

After solving for `hat(theta)_k`, construct

$$
 A_k^{id} = A(\hat{\theta}_k),
 \quad
 B_k^{id} = B(\hat{\theta}_k).
$$

Define corrections

$$
 \Delta A_k = A_k^{id} - A_0,
 \quad
 \Delta B_k = B_k^{id} - B_0.
$$

Then apply safety checks such as

$$
 \|\Delta A_k\|_F \le \delta_A,
 \quad
 \|\Delta B_k\|_F \le \delta_B.
$$

If invalid, non-finite, or too large:

- shrink the correction, or
- reuse the previous identified model, or
- fall back to nominal

The exact fallback should be logged clearly in the result bundle.

## 5.8 RL-controlled blend

In this new workflow, RL does **not** output plant inputs.
It outputs a single scalar blend factor in phase 1.

Let actor output be

$$
 a_k \in [-1, 1].
$$

Map to raw blend factor

$$
 \eta_k^{raw} = \frac{a_k + 1}{2} \in [0,1].
$$

Then smooth it:

$$
 \eta_k = (1 - \tau_\eta) \eta_{k-1} + \tau_\eta \eta_k^{raw}.
$$

Build the blended physical prediction model:

$$
 A_k^{pred} = A_0 + \eta_k (A_k^{id} - A_0),
$$

$$
 B_k^{pred} = B_0 + \eta_k (B_k^{id} - B_0).
$$

Then rebuild the augmented prediction model for MPC only:

$$
 A_{aug,k}^{pred} = \begin{bmatrix} A_k^{pred} & 0 \\ 0 & I \end{bmatrix},
 \quad
 B_{aug,k}^{pred} = \begin{bmatrix} B_k^{pred} \\ 0 \end{bmatrix}.
$$

This is the model used by the assisted MPC optimization.
The observer must not use it.

## 5.9 Reward

For phase 1, keep the stage reward aligned with the current MPC-style reward logic already used in unified runners.

Use the same scaled deviation tracking and move penalty structure:

$$
 r_k = -\left(e_{y,k}^T Q e_{y,k} + \Delta u_k^T R \Delta u_k\right)
$$

with the repo’s existing reward hooks reused as much as possible.

Do not redesign the reward in phase 1 unless absolutely necessary.

## 6. Repo-specific runtime design

## 6.1 Follow the `utils/matrix_runner.py` pattern

This new workflow should mirror the structure of `utils/matrix_runner.py`.

That means the runner should:

1. read config and runtime context from the notebook
2. build disturbance schedule using shared helpers if needed
3. assemble RL state using shared state-feature utilities
4. keep `A_base`, `B_base` as nominal prediction baseline
5. keep `A_est`, `B_est`, `L_nom` as fixed nominal observer model
6. update only the prediction model inside the MPC solve
7. return a normalized result bundle with logs and diagnostics

## 6.2 Recommended new runner module

Create a new module:

- `utils/reid_batch_runner.py`

This runner should **not** overwrite or mutate the current matrix runner.

Use the same result-bundle style as the current unified runners.

## 6.3 Recommended new notebook

Create a new notebook entrypoint:

- `RL_assisted_MPC_reid_batch_unified.ipynb`

This should be the polymer phase-1 launcher for the new workflow.

It should follow the same notebook style as the current unified family.

## 7. Algorithm to implement

## Initialization

1. Load nominal polymer system data using the current shared loader pipeline.
2. Build nominal physical and augmented models.
3. Build nominal observer gain using `utils/observer.py`.
4. Initialize observer state estimate `xhatdhat[:, 0]`.
5. Initialize identification buffer.
6. Set `theta_hat_0 = 0`.
7. Set `(A_id_0, B_id_0) = (A_0, B_0)`.
8. Set `eta_0 = 0.0` so the run begins exactly as nominal MPC.
9. Set prediction model to nominal.

## Each control step `k`

1. Build the RL state from the current nominal observer estimate and identification diagnostics.
2. Query the actor to get raw action `a_k`.
3. Map and smooth to `eta_k`.
4. Build the blended prediction model `(A_pred_k, B_pred_k)` using `(A_0, B_0)` and `(A_id_{k-1}, B_id_{k-1})`.
5. Rebuild the augmented prediction model for MPC only.
6. Solve MPC using the blended prediction model.
7. Apply the first control move to the plant.
8. Step the plant, including disturbance schedule if enabled.
9. Update the **nominal observer only** using the fixed nominal augmented model and `L_nom`.
10. Extract the physical state blocks `xhat_phys_k`, `xhat_phys_{k+1}`.
11. Push `(xhat_phys_k, u_dev_k, xhat_phys_{k+1})` into the rolling identification buffer.
12. Every `id_update_period` steps, if enough samples are available:
    - assemble the regression system
    - solve batch ridge or bounded LS
    - reconstruct `(A_id_k, B_id_k)`
    - apply safety checks and fallback if needed
13. Compute reward using the current shared reward hook.
14. Store the RL transition.
15. Train the RL agent after warm start using the existing agent APIs.

## 8. RL state design in this repo

Use the current shared state-feature layer as the base.

### Preferred implementation

Add a new helper in `utils/state_features.py` or a new dedicated helper in `utils/reid_batch.py` that builds a blend-policy state.

At minimum include:

1. current augmented nominal observer state `xhatdhat[:, k]`
2. current setpoint in scaled deviation coordinates
3. previous input in scaled deviation coordinates
4. previous blend factor `eta_{k-1}`
5. identification residual norm
6. correction magnitudes `||Delta A||_F` and `||Delta B||_F`
7. fallback / invalid-solve indicator

If this fits naturally as a new `state_mode`, implement it as such while keeping backward compatibility.

If not, add a separate helper rather than forcing the current builder to do too much.

## 9. What **not** to do

1. Do not build this on `Simulation/rl_sim.py`.
2. Do not replace the current matrix runner.
3. Do not let the observer use the re-identified or blended model.
4. Do not identify the full augmented model.
5. Do not make RL output plant inputs in this new workflow.
6. Do not break the current unified notebooks.
7. Do not push new logic into `BasicFunctions/` unless strictly necessary.

## 10. Logging and result bundle requirements

Match the current unified-runner style.

The result bundle must log at least:

- `eta_log`
- raw actor action
- `theta_hat_log` or periodic theta history
- batch fit residual norm
- `A_model_delta_ratio_log`
- `B_model_delta_ratio_log`
- fallback flags
- invalid solve count
- `delta_y_storage`
- `delta_u_storage`
- rewards and average rewards
- observer trajectory `xhatdhat`
- predicted outputs `yhat`
- `estimator_mode = "fixed_nominal"`
- `prediction_model_mode = "online_reid_blend"`

Keep naming consistent with existing runner bundles when possible.

## 11. Validation requirements

Implement or demonstrate the following checks.

### 11.1 Critical invariance tests

1. **Eta-zero baseline**
   - force `eta = 0` for the entire run
   - verify the new workflow reproduces nominal MPC behavior to numerical tolerance

2. **Observer isolation**
   - verify that changing the prediction model does not alter the observer model used in the update
   - the observer must always use the fixed nominal matrices and `L_nom`

3. **No-identification fallback**
   - if the identification module is disabled or has insufficient samples, the workflow should reduce to nominal MPC

### 11.2 Identification sanity checks

4. **Synthetic recovery test**
   - generate synthetic data from a known theta in the chosen basis family
   - verify batch ridge approximately recovers theta under light noise

5. **Bounds test**
   - verify theta stays within configured bounds after solve / clipping

6. **Invalid-model fallback**
   - non-finite or overly large corrections must trigger fallback and be logged

### 11.3 Polymer smoke tests

7. **Short polymer run**
   - new notebook / runner completes a short run successfully

8. **Comparison run**
   - compare nominal MPC vs the new workflow with plots for outputs, inputs, reward, eta, and model-delta metrics

## 12. Suggested file-level tasks for Codex

### 12.1 Add `utils/reid_batch.py`

Implement:

- `RollingIDBuffer`
- basis builder for polymer phase 1
- feature assembly
- ridge solver
- optional bounded LS solver
- model reconstruction
- safety checks and fallback helpers

### 12.2 Add `utils/reid_batch_runner.py`

Implement a runner similar in style to `utils/matrix_runner.py`.

It should:

- accept `reid_cfg` and `runtime_ctx`
- use current shared scaling and disturbance helpers
- keep fixed nominal observer logic
- run MPC with the blended prediction model
- produce a normalized result bundle

### 12.3 Add `RL_assisted_MPC_reid_batch_unified.ipynb`

Notebook responsibilities:

- build polymer system
- load data and scaling artifacts
- configure MPC, reward, and RL agent
- call the new runner
- save results and plots in the same style as the current unified notebook family

### 12.4 Extend shared state features only if needed

If a new RL state mode is necessary, add it carefully and keep current modes intact.

### 12.5 Add report file

Create:

- `report/05_online_reidentification_batch_ridge_blend.tex`

Also add any bibliography helper or `.bib` file if appropriate.

### 12.6 Add short change note

Because `AGENTS.md` asks for it for major changes, add a brief note under:

- `change-reports/`

## 13. Required LaTeX report contents

The LaTeX report must be extensive and math-heavy.

Include:

1. motivation for online re-identification + RL blend
2. why the current matrix workflow is related but not the same method
3. current repo architecture and why the unified runner layer is the correct implementation surface
4. explicit statement that the observer stays nominal
5. nominal physical and augmented model equations
6. physical-only identification logic
7. basis-family definition for phase 1 polymer
8. batch ridge derivation
9. optional bounded least-squares derivation
10. construction of identified model
11. RL blend derivation
12. construction of augmented prediction model for MPC only
13. full step-by-step algorithm
14. mapping from equations to repo files
15. diagnostics and validation plan
16. limitations and next steps, including later extension to other re-identification methods

## 14. References to include in the report

Use these as the core references:

1. Ljung, *System Identification: Theory for the User*, 2nd ed., 1999.
2. Hoerl and Kennard, "Ridge Regression: Biased Estimation for Nonorthogonal Problems," *Technometrics*, 1970.
3. Rawlings, Mayne, and Diehl, *Model Predictive Control: Theory, Computation, and Design*, 2nd ed., 2022.
4. Astrom and Wittenmark, *Adaptive Control*, 2nd ed., 1995.
5. Sutton and Barto, *Reinforcement Learning: An Introduction*, 2nd ed., 2018.

These are conceptual anchors. The implementation does not need to replicate any one paper exactly.

## 15. Acceptance criteria

This task is complete only if all of the following are true:

1. Existing unified workflows still run unchanged.
2. A new polymer-only unified workflow exists for online batch re-identification + RL blend.
3. The implementation lives on the current unified surface, not the stale legacy loop.
4. RL controls the blend factor `eta`, not the plant input, in the new workflow.
5. Identification uses only the physical state block in scaled deviation coordinates.
6. The observer is guaranteed to remain nominal.
7. A detailed LaTeX report is created under `report/`.
8. A short change note is added under `change-reports/`.
9. The `eta = 0` test reproduces nominal MPC to numerical tolerance.
10. The result bundle logs enough diagnostics to debug the new method.