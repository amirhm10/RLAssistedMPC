# Codex Plan: Polymer-Only Prediction-Error-Validated Markov Correction for Offset-Free MPC

Use the `$research-result-loop` skill for this task.

## High-level instruction

Implement and document a polymer-only test notebook for a new method:

**Prediction-error-validated Markov-parameter correction for offset-free MPC**

The goal is to test whether correcting the finite-horizon input-output Markov map improves prediction and control behavior before modifying any shared RL/MPC code.

This is a research prototype. Do not modify existing source modules. Add new files only.

## Strict preservation rules

1. Do not edit existing files in:
   - `Simulation/`
   - `utils/`
   - `TD3Agent/`
   - `SACAgent/`
   - `DQN/`
   - `DuelingDQN/`
   - `systems/`
   - existing root notebooks

2. You may import and use existing functions/classes.

3. All new implementation code for this first pass must live inside a new notebook and optional new report scripts under `report/scripts/`.

4. Do not overwrite old results or figures.

5. Save all new outputs under dated folders.

6. Do not claim the method works unless the metrics support it.

7. Keep all math in LaTeX in the report.

8. Keep this first implementation polymer-only. Do not touch distillation files.

## Required new files

Create these new files:

1. New notebook at repo root:

```text
polymer_markov_corrected_mpc_unified.ipynb
```

2. New living report:

```text
report/polymer_markov_correction_progress.md
```

3. New analysis/output folder:

```text
report/figures/polymer_markov_correction_YYYYMMDD/
```

4. Optional helper script only if useful for reproducible report asset generation:

```text
report/scripts/generate_polymer_markov_correction_assets.py
```

Do not edit existing report files unless absolutely necessary. The main progress tracking must be in the new report file above.

## Existing repo context to use

Use the existing polymer setup and patterns from the repo.

Relevant existing files to inspect and reuse:

- `Simulation/mpc.py`
  - `MpcSolverGeneral`
  - `augment_state_space`
  - `compute_observer_gain`
  - existing MPC objective structure
- `utils/helpers.py`
  - scaling helpers and setpoint generation utilities if needed
- `utils/observer.py`
  - observer update utilities if needed
- `utils/rewards.py`
  - reward functions if needed
- existing polymer notebooks:
  - `MPCOffsetFree_unified.ipynb`
  - `RL_assisted_MPC_matrices_unified.ipynb`
  - `RL_assisted_MPC_structured_matrices_unified.ipynb`
  - `RL_assisted_MPC_residual_unified.ipynb`
- existing data/results:
  - `Polymer/Data/`
  - `Polymer/Results/`
- existing reports:
  - `report/matrix_multiplier_progress_summary_2026_04_28.md`
  - `report/matrix_multiplier_cap_calculation_and_distillation_recovery.md`
  - `report/figures/matrix_multiplier_step3d_20260430/polymer_step3d_latest_summary.csv`

Important lesson from existing reports:

- Step 3D-style hard usefulness gating did not work in polymer.
- The hard gate accepted no candidates.
- Executed multipliers stayed nominal.
- The method collapsed to near-MPC behavior.
- Therefore, do not make internal MPC-cost comparison the primary acceptance criterion.

The new method must instead use recent plant prediction error as the primary evidence.

## Scientific objective

Current matrix-multiplier method changes the model as:

\[
A_t = \alpha_t A_0
\]

\[
B_t = B_0 \operatorname{diag}(\delta_t)
\]

This is restrictive because it changes all finite-horizon input-output effects through a small number of state-space multipliers.

The proposed method keeps the offset-free MPC structure but corrects the finite-horizon input-output map:

\[
Y_k =
Y_{\mathrm{free},k}
+
G_z(P,M)\Delta U_k
\]

where \(G_z(P,M)\) is built from corrected Markov blocks:

\[
M_i(z_k)
=
M_{i,0}
+
\sum_{j=1}^{r}z_{j,k}M_{i,j}^{\mathrm{basis}}
\]

The correction must first prove that it reduces recent plant prediction error:

\[
S_{\mathrm{pred}}(z)
=
\sum_{\tau=k-W}^{k-1}
\left(
\|W_yE_0(\tau)\|_2^2
-
\|W_yE_z(\tau)\|_2^2
\right)
-
\lambda_z\|z\|_2^2
\]

where:

\[
E_0(\tau)=Y_\tau^{\mathrm{meas}}-Y_\tau^0
\]

\[
E_z(\tau)=Y_\tau^{\mathrm{meas}}-Y_\tau^z
\]

The main criterion is:

\[
S_{\mathrm{pred}}(z)>S_{\min}
\]

MPC internal cost comparison is only a loose safety guard, not the main usefulness gate.

## Method mathematics

### 1. Nominal offset-free model

The nominal discrete model is:

\[
x_{k+1}=Ax_k+B\Delta u_k
\]

\[
y_k=Cx_k+d_k
\]

The offset-free augmented model is:

\[
x_{a,k}
=
\begin{bmatrix}
x_k\\
d_k
\end{bmatrix}
\]

\[
A_a=
\begin{bmatrix}
A & 0\\
0 & I
\end{bmatrix}
\]

\[
B_a=
\begin{bmatrix}
B\\
0
\end{bmatrix}
\]

\[
C_a=
\begin{bmatrix}
C & I
\end{bmatrix}
\]

The observer provides:

\[
\hat{x}_{a,k}
=
\begin{bmatrix}
\hat{x}_k\\
\hat{d}_k
\end{bmatrix}
\]

Keep the observer nominal in this first implementation.

### 2. Markov blocks

For the nominal physical part:

\[
M_i = C A^{i-1} B
\]

For the augmented model, because the disturbance states do not respond to future input moves, the forced-response blocks are equivalent to:

\[
M_i = C_{\mathrm{phys}}A_{\mathrm{phys}}^{i-1}B_{\mathrm{phys}}
\]

or equivalently:

\[
M_i = C_a A_a^{i-1}B_a
\]

Both should match for the forced response.

### 3. Lifted prediction

For prediction horizon \(P\) and control horizon \(M\):

\[
\Delta U_k =
\begin{bmatrix}
\Delta u_{k|k}\\
\Delta u_{k+1|k}\\
\vdots\\
\Delta u_{k+M-1|k}
\end{bmatrix}
\]

\[
Y_k =
\begin{bmatrix}
y_{k+1|k}\\
y_{k+2|k}\\
\vdots\\
y_{k+P|k}
\end{bmatrix}
\]

The lifted prediction is:

\[
Y_k=Y_{\mathrm{free},k}+G_0(P,M)\Delta U_k
\]

where:

\[
Y_{\mathrm{free},k}=
\begin{bmatrix}
CA\hat{x}_k+\hat{d}_k\\
CA^2\hat{x}_k+\hat{d}_k\\
\vdots\\
CA^P\hat{x}_k+\hat{d}_k
\end{bmatrix}
\]

and:

\[
G_0(P,M)=
\begin{bmatrix}
M_1 & 0 & 0 & \cdots\\
M_2 & M_1 & 0 & \cdots\\
M_3 & M_2 & M_1 & \cdots\\
\vdots & \vdots & \vdots & \ddots\\
M_P & M_{P-1} & M_{P-2} & \cdots
\end{bmatrix}
\]

When \(j\ge M\), use the last control move convention consistently with `MpcSolverGeneral.mpc_opt_fun`, which uses:

```python
idx = j if j < self.NC else self.NC - 1
```

This means after the control horizon, the last optimized input level is held constant.

Important implementation detail:

In the existing `MpcSolverGeneral`, `U` appears to be absolute input deviation sequence, not move increments. The objective penalizes moves by:

\[
\Delta u_j = U_j - U_{j-1}
\]

where \(U_{-1}=u_{\mathrm{prev,dev}}\).

Therefore, in the lifted prediction, use the same convention:

\[
U =
\begin{bmatrix}
u_{k|k}^{\mathrm{dev}}\\
u_{k+1|k}^{\mathrm{dev}}\\
\vdots
\end{bmatrix}
\]

not raw \(\Delta u\), unless the notebook explicitly converts between the two.

To avoid mismatch with existing MPC, first implement the lifted prediction using the same `U` convention as `MpcSolverGeneral`.

### 4. Corrected Markov blocks

Define:

\[
M_i(z)=M_{i,0}+\sum_{j=1}^{r}z_jM_{i,j}^{\mathrm{basis}}
\]

Start with small candidate basis families:

#### Basis family A: input-output pair gain correction

\[
[M_i(z)]_{p,q}
=
[M_{i,0}]_{p,q}(1+z_{p,q})
\]

For polymer with two outputs and two inputs:

\[
z=
[z_{1,1},z_{1,2},z_{2,1},z_{2,2}]
\]

Use small bounds such as:

\[
z_{p,q}\in[-0.05,0.05]
\]

#### Basis family B: input-channel gain correction

\[
M_i(z)=M_{i,0}\operatorname{diag}(1+z_1,1+z_2)
\]

This is closer to B-column scaling, but directly in Markov space.

#### Basis family C: delay-shift correction

Define:

\[
M_i^{\mathrm{delay}}=M_{i-1,0}
\]

with \(M_0=0\).

Then:

\[
M_i(z)=M_{i,0}+z_{\mathrm{delay}}(M_i^{\mathrm{delay}}-M_{i,0})
\]

Use small bounds such as:

\[
z_{\mathrm{delay}}\in[0,0.3]
\]

Do not use all bases at once in the first run. Make them selectable by config.

### 5. Prediction-error validation

For a historical window \(\tau=k-W,\dots,k-1\), build measured output stacks:

\[
Y_\tau^{\mathrm{meas}}=
\begin{bmatrix}
y_{\tau+1}\\
\vdots\\
y_{\tau+P}
\end{bmatrix}
\]

Build nominal predictions:

\[
Y_\tau^0=Y_{\mathrm{free},\tau}+G_0(P,M)U_\tau^{\mathrm{real}}
\]

Build corrected predictions:

\[
Y_\tau^z=Y_{\mathrm{free},\tau}+G_z(P,M)U_\tau^{\mathrm{real}}
\]

Then:

\[
S_{\mathrm{pred}}(z)
=
\sum_{\tau=k-W}^{k-1}
\left(
\|W_y(Y_\tau^{\mathrm{meas}}-Y_\tau^0)\|_2^2
-
\|W_y(Y_\tau^{\mathrm{meas}}-Y_\tau^z)\|_2^2
\right)
-
\lambda_z\|z\|_2^2
\]

Use \(W_y\) to normalize output scales. Use scaled deviation coordinates first if available, because the existing MPC and reward logic use scaled deviation variables.

### 6. Loose safety guard

Only after prediction-error validation, use a loose safety guard:

\[
g_z(P,M)
=
\frac{
\|W_y(G_z(P,M)-G_0(P,M))W_u\|_F
}{
\|W_yG_0(P,M)W_u\|_F+\epsilon
}
\]

Require:

\[
g_z(P,M)\le g_{\max}
\]

Also compute nominal-cost safety only as a loose catastrophe guard:

\[
J_0(U_z^\star)
\le
J_0(U_0^\star)+\epsilon_{\mathrm{loose}}
\]

Do not make this as strict as Step 3D.

## Required notebook design

Create a new notebook:

```text
polymer_markov_corrected_mpc_unified.ipynb
```

The notebook must have the following sections.

### Section 1: Imports and configuration

Create a config dictionary at the top.

Example fields:

```python
CONFIG = {
    "run_mode": "disturb",
    "n_tests": 50,
    "set_points_len": 200,
    "warm_start": 5,
    "predict_h": 6,
    "cont_h": 3,
    "basis_family": "io_pair_gain",
    "z_bound": 0.05,
    "prediction_window": 20,
    "lambda_z": 1e-3,
    "s_pred_min": 1e-6,
    "gain_drift_max": 0.10,
    "nominal_cost_relative_tol": 0.10,
    "nominal_cost_absolute_tol": 1e-8,
    "run_shadow_only": True,
    "run_adaptive_ls": True,
    "run_live_corrected_mpc": True,
    "run_rl_proposal": False,
    "save_outputs": True,
    "make_plots": True,
}
```

Notes:

- Keep `run_shadow_only=True` available.
- The user requested implementing Phase 1 to 5 together in one pass, so include all phase logic in the notebook.
- Do not make RL mandatory. Include Phase 5 scaffolding or a local simple proposal policy, but keep the first working result focused on non-RL prediction-error validation and adaptive correction.
- If implementing live correction is too risky, keep it behind config and default it to `False`. But the code path should exist.

### Section 2: Load polymer model and baseline objects

Use existing repo data and notebook patterns.

Load:

- polymer `system_dict`
- scaling factors
- steady states
- input/output min-max
- baseline MPC setup

If paths differ, discover them from existing polymer notebooks and use robust path checks.

The notebook should print:

- shape of \(A\)
- shape of \(B\)
- shape of \(C\)
- \(n_x,n_u,n_y\)
- prediction horizon
- control horizon
- input bounds
- steady-state input
- steady-state output

### Section 3: Local Markov helper functions

Implement these functions locally inside the notebook:

```python
def compute_markov_blocks(A, B, C, P_max):
    ...

def build_toeplitz_from_markov(M_blocks, P, M, hold_last=True):
    ...

def free_response(A, C, x0, dhat, P):
    ...

def lifted_predict(A, C, x0, dhat, G, U_seq, P, M):
    ...

def state_space_predict(A_aug, B_aug, C_aug, x0_aug, U_seq, P, M):
    ...

def make_markov_basis(M_blocks, basis_family):
    ...

def apply_markov_correction(M_blocks, basis_blocks, z):
    ...

def gain_drift(Gz, G0, Wy=None, Wu=None, eps=1e-12):
    ...

def prediction_improvement_score(...):
    ...
```

Keep these local in the notebook.

Do not add them to `utils/` yet.

### Section 4: Phase 1, lifted-prediction equivalence validation

This is mandatory.

For randomly sampled \(x_0\) and \(U\), verify:

\[
Y_{\mathrm{state-space}} \approx Y_{\mathrm{lifted}}
\]

Save metrics:

```text
max_abs_error
mean_abs_error
relative_error
```

Plot:

- state-space predicted outputs versus lifted predicted outputs
- prediction error by horizon step

Save figure under:

```text
report/figures/polymer_markov_correction_YYYYMMDD/phase1_lifted_equivalence.png
```

Pass criterion:

```text
max_abs_error < 1e-8
```

If not passed, do not continue to live correction.

### Section 5: Phase 2, shadow candidate prediction-error scoring

Use real or simulated nominal MPC polymer rollout data.

For each step after enough history exists:

1. Build nominal prediction stacks.
2. Build candidate corrections \(z^{(j)}\).
3. Compute \(S_{\mathrm{pred}}(z^{(j)})\).
4. Do not execute corrected MPC in this phase.
5. Log the best candidate and its score.

Candidate library examples:

```python
candidate_z = [
    np.zeros(r),
    +0.02 * e_j,
    -0.02 * e_j,
    +0.05 * e_j,
    -0.05 * e_j,
]
```

For delay basis, include:

```python
z_delay = [0.05, 0.10, 0.20]
```

Metrics:

- fraction of time any candidate has \(S_{\mathrm{pred}}>0\)
- mean best \(S_{\mathrm{pred}}\)
- output-wise prediction error improvement
- candidate selection frequency
- gain drift of selected candidate

Figures:

1. Best prediction-improvement score over time.
2. Candidate selection histogram.
3. Nominal versus corrected one-step or multi-step prediction error.
4. Output-wise prediction improvement.

### Section 6: Phase 3, adaptive constrained LS correction

Implement regularized least-squares estimation of \(z_k\):

\[
z_k^\star
=
\arg\min_z
\sum_{\tau=k-W}^{k-1}
\left\|
W_y
\left(
Y_\tau^{\mathrm{meas}}
-
Y_{\mathrm{free},\tau}
-
G_z(P,M)U_\tau^{\mathrm{real}}
\right)
\right\|_2^2
+
\lambda_z\|z\|_2^2
\]

Subject to:

\[
z_{\min}\le z\le z_{\max}
\]

and optionally:

\[
\|z-z_{k-1}\|_\infty\le\Delta z_{\max}
\]

Use `scipy.optimize.minimize` with bounds first.

Do not rely on CVXPY for this prototype unless it is already used and available.

Metrics:

- \(z_k^\star\) trace
- \(S_{\mathrm{pred}}(z_k^\star)\)
- accepted fraction based on prediction-error validation
- gain drift trace
- regularization penalty trace

Figures:

1. \(z_k^\star\) over time.
2. Accepted versus rejected correction steps.
3. Prediction error before and after LS correction.
4. Gain drift over time.

### Section 7: Phase 4, corrected MPC execution with loose safety

Add a local lifted-MPC objective in the notebook:

```python
def lifted_mpc_cost(U_flat, y_sp, u_prev_dev, Y_free, G, Q_out, R_in, P, M):
    ...
```

Important:

Match the existing `MpcSolverGeneral.mpc_opt_fun` convention:

- optimization variable is input-deviation sequence \(U_j\), not raw moves
- move penalty uses \(U_j-U_{j-1}\)
- after control horizon, hold last input sequence

Then implement two solves per step:

1. nominal lifted MPC using \(G_0\)
2. corrected lifted MPC using \(G_z\)

Execution logic:

```text
if S_pred(z) > S_min and gain_drift <= gain_drift_max and nominal_cost_is_not_catastrophic:
    execute corrected first input
else:
    execute nominal first input
```

The nominal-cost guard is:

\[
J_0(U_z^\star)
\le
J_0(U_0^\star)
+
\epsilon_{\mathrm{loose}}
\]

with:

\[
\epsilon_{\mathrm{loose}}
=
\epsilon_{\mathrm{abs}}
+
\epsilon_{\mathrm{rel}}|J_0(U_0^\star)|
\]

Start with a loose relative tolerance such as `0.10`.

Do not use the old Step 3D hard gate thresholds.

Metrics:

- corrected-execution acceptance rate
- fallback rate
- average reward compared with nominal MPC
- output MAE
- input movement
- constraint hits
- \(S_{\mathrm{pred}}\) on accepted versus rejected steps
- gain drift on accepted versus rejected steps

Figures:

1. Output trajectories, nominal MPC versus Markov-corrected MPC.
2. Input trajectories, nominal MPC versus Markov-corrected MPC.
3. Episode reward comparison.
4. Accepted/rejected correction markers.
5. \(z\) traces and gain-drift trace.
6. Prediction-error improvement versus reward improvement.

### Section 8: Phase 5, RL proposal scaffold

Implement Phase 5 as a notebook-local scaffold.

Do not modify `TD3Agent`.

The Phase 5 options can be:

#### Option 5A: local heuristic policy

Use a simple proposal policy based on recent best LS correction:

\[
z_k^{\mathrm{proposal}}=z_k^\star
\]

This is not RL, but it creates the same proposal, gate, execution, and replay data format.

#### Option 5B: existing TD3 agent used locally

If practical, instantiate an existing `TD3Agent` in the notebook with:

- state = observer state, tracking error, innovation, previous input, previous \(z\)
- action = \(z\)
- reward = existing polymer reward or physical tracking reward

But keep this optional and behind:

```python
CONFIG["run_rl_proposal"]
```

If enabled:

- store executed \(z\), not requested \(z\), in replay
- do not change `TD3Agent`
- do not change replay buffer code
- log requested \(z\), executed \(z\), and fallback status

It is acceptable for this first pass to implement the RL data interface and leave full TD3 training disabled by default, as long as the notebook clearly contains the phase and report explains what remains.

### Section 9: Save result bundle

Save a pickle under a new folder:

```text
Polymer/Results/polymer_markov_corrected_mpc/YYYYMMDD_HHMMSS/input_data.pkl
```

Include:

```python
result_bundle = {
    "method": "prediction_error_validated_markov_correction",
    "config": CONFIG,
    "A": A,
    "B": B,
    "C": C,
    "A_aug": A_aug,
    "B_aug": B_aug,
    "C_aug": C_aug,
    "M_blocks_nominal": M_blocks,
    "basis_family": basis_family,
    "z_log": z_log,
    "z_proposed_log": z_proposed_log,
    "z_executed_log": z_executed_log,
    "s_pred_log": s_pred_log,
    "gain_drift_log": gain_drift_log,
    "accepted_log": accepted_log,
    "fallback_log": fallback_log,
    "y_nominal": y_nominal,
    "u_nominal": u_nominal,
    "y_markov": y_markov,
    "u_markov": u_markov,
    "rewards_nominal": rewards_nominal,
    "rewards_markov": rewards_markov,
    "avg_rewards_nominal": avg_rewards_nominal,
    "avg_rewards_markov": avg_rewards_markov,
    "prediction_error_nominal_log": prediction_error_nominal_log,
    "prediction_error_markov_log": prediction_error_markov_log,
    "figures": list_of_saved_figure_paths,
}
```

Also save CSV summaries:

```text
report/figures/polymer_markov_correction_YYYYMMDD/summary_metrics.csv
report/figures/polymer_markov_correction_YYYYMMDD/prediction_score_summary.csv
report/figures/polymer_markov_correction_YYYYMMDD/acceptance_summary.csv
```

## Required report content

Create or update:

```text
report/polymer_markov_correction_progress.md
```

Use the `$research-result-loop` style and include these sections:

1. Objective
2. Files inspected
3. What the existing method was doing
4. Why Step 3D is not reused as the main gate
5. Mathematical formulation
6. Phase 1: lifted-prediction equivalence validation
7. Phase 2: shadow prediction-error scoring
8. Phase 3: adaptive LS Markov correction
9. Phase 4: corrected MPC with loose safety guard
10. Phase 5: RL proposal scaffold
11. Result summary
12. Figures
13. Bugs, inconsistencies, or risks found
14. Limitations
15. Next experiment
16. Remaining uncertainty

In the report, explicitly explain the key methodological difference:

Old Step 3D tested:

\[
J_0(U_z^\star)
\quad \text{and} \quad
J_z(U_0^\star)-J_z(U_z^\star)
\]

The adjusted Markov method first tests:

\[
\|Y^{\mathrm{meas}}-Y^0\|^2
-
\|Y^{\mathrm{meas}}-Y^z\|^2
\]

The report must include figures. At minimum:

1. `phase1_lifted_equivalence.png`
2. `phase2_prediction_score_trace.png`
3. `phase2_candidate_selection_histogram.png`
4. `phase3_z_trace.png`
5. `phase3_prediction_error_improvement.png`
6. `phase4_outputs_compare.png`
7. `phase4_inputs_compare.png`
8. `phase4_reward_compare.png`
9. `phase4_acceptance_and_gain_drift.png`

If a figure cannot be generated, write why in the report.

## Required verification checks

At the end of the notebook, print a verification table:

```text
Check | Value | Pass?
Lifted equivalence max error | ... | yes/no
Any positive shadow S_pred fraction | ... | yes/no
Adaptive LS accepted fraction | ... | yes/no
Live corrected accepted fraction | ... | yes/no
Reward delta mean | ... | yes/no
Output MAE delta | ... | yes/no
Input movement delta | ... | yes/no
```

Also save this table as CSV.

## Success criteria

The method should not be declared successful unless these are true:

1. Lifted prediction equivalence passes.
2. At least one candidate or adaptive correction gives positive prediction-error improvement on a meaningful fraction of steps.
3. The corrected MPC does not simply collapse to nominal MPC unless the prediction-error test rejects the correction.
4. If live correction is enabled, output MAE does not degrade materially.
5. Input movement does not increase excessively.
6. Any reward improvement is supported by physical tracking metrics, not reward alone.

## Failure criteria

The method should be considered not promising if:

1. Lifted prediction does not match state-space prediction.
2. No candidate correction improves recent prediction error.
3. LS correction saturates at bounds most of the time.
4. Prediction improvement exists but closed-loop tracking gets worse.
5. The loose safety guard rejects almost everything.
6. The method only improves one output while strongly worsening the other output or input movement.

## Suggested implementation order for Codex

1. Inspect relevant existing polymer notebook and `Simulation/mpc.py`.
2. Create `polymer_markov_corrected_mpc_unified.ipynb`.
3. Add local Markov helper cells.
4. Add Phase 1 equivalence validation.
5. Add Phase 2 shadow scoring.
6. Add Phase 3 adaptive LS correction.
7. Add Phase 4 corrected lifted-MPC execution behind config.
8. Add Phase 5 RL proposal scaffold behind config.
9. Save result bundle and CSV summaries.
10. Create figures under a dated report figure folder.
11. Create `report/polymer_markov_correction_progress.md`.
12. Add a short change report under `change-reports/` only if that is consistent with repo practice.

## Important coding details

### Matching existing MPC convention

In `Simulation/mpc.py`, `MpcSolverGeneral.mpc_opt_fun` uses:

```python
U = x[:n_inputs * self.NC].reshape(self.NC, n_inputs)
...
idx = j if j < self.NC else self.NC - 1
x_pred[:, j + 1] = self.A @ x_pred[:, j] + self.B @ U[idx, :]
...
U_prev = np.vstack([u_prev_dev.reshape(1, -1), U[:-1, :]])
du = U - U_prev
```

Therefore, the notebook lifted objective must treat the optimized sequence as input deviation levels, not raw input increments.

### Scaled versus physical variables

Use scaled deviation coordinates for MPC and Markov prediction unless the notebook explicitly converts to physical units for plotting.

Track physical metrics separately for report readability.

### Observer

Keep observer nominal for this prototype.

Do not update observer gain based on corrected Markov blocks.

### Control horizon

Use the same hold-last convention as existing MPC:

\[
U_j=U_{M-1}
\quad \text{for} \quad j\ge M
\]

### Baseline comparison

Run or load a nominal MPC baseline using the same setpoints, same disturbances, same seed, same horizon, and same reward.

Do not compare against mismatched episode lengths or different disturbance profiles.

## Final response expected from Codex

When done, provide:

1. Files created
2. Notebook path
3. Report path
4. Figure folder path
5. Result bundle path
6. Verification table
7. Main result summary
8. Whether the method passed or failed the first-pass criteria
9. What to test next

Do not say the method works unless the evidence supports it.
