# Codex plan for fixing the current online re-identification + RL blend issues in `RLAssistedMPC`

## 1. Purpose

This task is a **debug-and-improve pass** for the current polymer online batch re-identification workflow.

Target workflow:

- `RL_assisted_MPC_reid_batch_unified.ipynb`
- `utils/reid_batch_runner.py`
- `utils/reid_batch.py`
- `systems/polymer/notebook_params.py`
- plotting helpers used by the re-identification notebook

The goal is **not** to rewrite the method from scratch.
The goal is to fix the specific issues observed in the current run:

1. rewards change only slightly overall
2. RL often learns to reduce `eta` toward zero instead of using the identified model
3. the identified model appears too weak and too constrained to help
4. the current diagnostics are not rich enough to explain whether the problem is basis choice, saturation, state design, or update timing

This must remain **additive** and **backward compatible**.

## 2. Observed issues from the current run

The current run suggests the following behavior:

1. Early episodes are nominal by design because warm start forces `eta = 0`.
2. After warm start, RL uses nonzero `eta`, but average reward often becomes worse.
3. Later, RL tends to drive `eta` back down, which suggests the learned policy is discovering that the identified model is not helping much.
4. The identified `B` correction is frequently very close to the configured cap, which suggests saturation / overconstraint.
5. The current phase-1 basis is extremely small and is still very close to the old matrix-multiplier logic.
6. The blend-policy extras are appended in raw scale, which likely weakens the RL state quality.
7. The nominal observer is correctly kept fixed, but the current observer correction timing should be ablated because it may be hurting identification quality.

This plan is specifically meant to fix those issues.

## 3. Non-negotiable rules

1. Keep the observer nominal.
   - Do not use the identified model or blended model in the observer update.
   - The observer must continue to use the nominal augmented model and `L_nom` only.

2. Keep the implementation additive.
   - Do not break existing matrix, residual, weights, horizon, or combined workflows.
   - Do not remove the current `reid_batch` workflow.

3. Use the current unified repo surface.
   - Work through `utils/` and the unified notebook family.
   - Do not move this work into stale legacy loops.

4. Polymer only for this pass.

## 4. High-level diagnosis to act on

The current implementation likely underperforms for four main reasons:

### 4.1 Basis family is too weak

The current basis in `utils/reid_batch.py` uses:

- one global correction direction for `A`
- one correction per input column of `B`

This is too close to the old matrix multiplier workflow.
It gives the identifier almost no expressive power.

### 4.2 Constraints are effectively double-clipping the identified model

The current workflow uses both:

- hard theta bounds
- relative Frobenius correction caps

This likely overconstrains the identified model and drives `B` to the edge of the admissible region.

### 4.3 Blend-policy state extras are not normalized

The appended diagnostics for the blend policy currently include raw values such as:

- previous `eta`
- residual norm
- `A` ratio
- `B` ratio
- valid flag

These should be normalized or transformed before being concatenated with the RL state.

### 4.4 Identification data may be hurt by observer-update timing

The observer remains nominal, which is correct, but the timing of the measurement correction should be ablated because it may reduce identification quality.

## 5. Required implementation strategy

Do **not** overwrite the current method blindly.

Instead, implement a **v2 improvement path** that preserves the current version for comparison.

Recommended additions:

- `utils/reid_batch.py`: extend with richer bases, richer diagnostics, and configurable safety modes
- `utils/reid_batch_runner.py`: extend with ablations and improved state features in a backward-compatible way
- `systems/polymer/notebook_params.py`: add a new `reid_batch_v2` family or a new config profile under the current reid family
- `RL_assisted_MPC_reid_batch_v2_unified.ipynb` or a clearly separated config path in the existing notebook
- updated plotting / diagnostics helpers
- updated report in `report/`
- short note under `change-reports/`

## 6. Concrete fixes to implement

## 6.1 Add richer identification bases

### 6.1.1 Keep the current basis as a legacy option

Do not remove the existing basis.
Rename it conceptually as:

- `basis_family = "scalar_legacy"`

This is needed for direct comparison.

### 6.1.2 Add a row/column basis as the new main option

Add a new basis family for the polymer physical model:

#### A model

Use row scaling:

$$
A(\theta^A) = A_0 + \sum_{i=1}^{n_x} \theta_i^A E_i,
\quad
E_i = e_i e_i^T A_0
$$

This allows each physical state equation to have its own correction strength.

#### B model

Use input-column scaling:

$$
B(\theta^B) = B_0 + \sum_{j=1}^{n_u} \theta_j^B F_j,
\quad
F_j = B_0 e_j e_j^T
$$

This is still structured and stable, but much richer than the current 3-parameter basis.

### 6.1.3 Optional block basis

If cleanly implementable, add a second richer structured basis:

- `basis_family = "block_polymer"`

based on a small block partition of the physical state vector.

This is optional. The required richer basis is the row/column family.

### 6.1.4 Config requirement

Add a basis selector in the config:

- `basis_family = "scalar_legacy" | "rowcol" | "block_polymer"`

For the new v2 path, default to:

- `basis_family = "rowcol"`

## 6.2 Fix overconstraint and saturation

### 6.2.1 Add configurable candidate-guard mode

The current logic uses both theta bounds and Frobenius caps. Make this explicit and configurable.

Add:

- `candidate_guard_mode = "theta_only" | "fro_only" | "both"`

Meaning:

- `theta_only`: rely on theta bounds only
- `fro_only`: use broad theta bounds but validate only through Frobenius caps
- `both`: use both

For the improved v2 path, default to:

- `candidate_guard_mode = "fro_only"`

or a clearly relaxed `both` profile

The main point is to stop the current double-restriction from silently pinning the identified model.

### 6.2.2 Add saturation diagnostics

Log the following at each identification update:

- `theta_unclipped`
- `theta_active`
- lower-bound hit mask
- upper-bound hit mask
- fraction of theta values clipped
- candidate `A` ratio and candidate `B` ratio before fallback
- active `A` ratio and active `B` ratio after fallback

This is necessary to confirm whether the identified model is saturating.

### 6.2.3 Relax the defaults for the v2 path

Do not blindly replace the current defaults. Add a new config profile for the improved workflow.

Recommended v2 direction:

- broader theta bounds than the current `[-0.05, 0.05]`
- slightly less restrictive `delta_A_max`, `delta_B_max`
- or use `fro_only` guard mode with broader theta bounds

Codex should implement the config structure and add reasonable research defaults, but preserve the old defaults for the legacy path.

## 6.3 Normalize the blend-policy state extras

### 6.3.1 Problem

The current `build_blend_policy_state(...)` appends raw extras to the state.
That is likely poor for learning because the appended values are on inconsistent scales.

### 6.3.2 Required fix

Add a normalized blend-state builder, for example:

- `build_blend_policy_state_v2(...)`

Use transformed and clipped features such as:

#### Eta

Map previous eta to symmetric scale:

$$
z_\eta = 2\eta_{k-1} - 1
$$

#### Residual norm

Use a stabilized transform such as:

$$
z_r = \mathrm{clip}\left(\frac{\log(1 + r_k)}{s_r}, -3, 3\right)
$$

or a running-normalized version.

#### Model delta ratios

Normalize relative to configured admissible caps:

$$
z_A = \mathrm{clip}\left(\frac{\|\Delta A_k\|/\|A_0\|}{\delta_A^{max}}, 0, 1.5\right)
$$

$$
z_B = \mathrm{clip}\left(\frac{\|\Delta B_k\|/\|B_0\|}{\delta_B^{max}}, 0, 1.5\right)
$$

#### Valid flag

Keep as 0/1.

### 6.3.3 Candidate-vs-active features

Also add the option to include:

- candidate `A` ratio
- candidate `B` ratio
- fallback-used flag
- ID-update-success flag

This gives RL better information about whether the identifier is trustworthy.

## 6.4 Improve identification diagnostics and ablations

### 6.4.1 Required ablation harness

The workflow already has useful knobs like `force_eta_constant` and `disable_identification`.
Extend the notebook and/or helper logic so the following runs are easy and standardized:

1. `eta = 0`, identification off
2. `eta = 0`, identification on
3. `eta = 1`, identification on
4. learned `eta`, identification on
5. learned `eta`, identification off

This will separate three questions:

- is the identified model itself useful?
- is RL blend helping or hurting?
- is the issue identification quality or policy learning?

### 6.4.2 Plotting upgrades

Update the plotting functions so the re-identification workflow automatically plots:

- `eta`
- candidate and active theta
- candidate and active model-delta ratios
- residual norm
- clipping counts or clipping masks
- fallback usage
- reward versus eta scatter or episode overlay

Without these plots, debugging remains too indirect.

## 6.5 Add observer-update timing ablation

### 6.5.1 Non-negotiable rule

The observer must remain nominal.
This must not change.

### 6.5.2 What to ablate

Add a config option for the re-identification workflow only:

- `observer_update_alignment = "legacy_previous_measurement" | "current_measurement"`

The goal is **not** to make the observer adaptive.
The goal is to test whether identification quality improves when the nominal observer update is aligned with the current measurement at the next-state estimate step.

### 6.5.3 Scope restriction

This ablation should be isolated to the re-identification workflow and must not silently change the behavior of the other unified runners.

## 6.6 Improve basis-aware defaults in notebook params

Add a new polymer notebook family or config profile, for example:

- `reid_batch_v2`

with explicit fields such as:

- `basis_family`
- `candidate_guard_mode`
- `observer_update_alignment`
- `normalize_blend_extras`
- `blend_extra_clip`
- `blend_residual_scale`
- `log_theta_clipping`
- `plot_candidate_vs_active`

The current `reid_batch` defaults should remain available for legacy comparison.

## 6.7 Keep the current method as a comparison baseline

The current implementation should remain runnable as:

- legacy re-identification baseline

The improved method should be a new path, not a destructive overwrite.

## 7. Mathematical guidance for the improved method

## 7.1 Identification model

For the row/column basis:

$$
A_k^{id} = A_0 + \sum_{i=1}^{n_x} \theta_{k,i}^A E_i,
\quad
E_i = e_i e_i^T A_0
$$

$$
B_k^{id} = B_0 + \sum_{j=1}^{n_u} \theta_{k,j}^B F_j,
\quad
F_j = B_0 e_j e_j^T
$$

Use the same physical-state rolling batch data:

$$
(\hat{x}_t^{phys}, u_t^{dev}, \hat{x}_{t+1}^{phys})
$$

and solve

$$
\hat{\theta}_k = \arg\min_{\theta}
\sum_{t=k-W}^{k-1} \|r_t - \Phi_t \theta\|_2^2
+ \lambda_{prev}\|\theta - \hat{\theta}_{k-1}\|_2^2
+ \lambda_0\|\theta\|_2^2
$$

with optional bounded least squares.

## 7.2 RL blend

Keep the same basic blend structure:

$$
\eta_k^{raw} = \frac{a_k + 1}{2},
\qquad
\eta_k = (1 - \tau_\eta)\eta_{k-1} + \tau_\eta \eta_k^{raw}
$$

$$
A_k^{pred} = A_0 + \eta_k(A_k^{id} - A_0)
$$

$$
B_k^{pred} = B_0 + \eta_k(B_k^{id} - B_0)
$$

Then augment for MPC prediction only.

## 8. Concrete file-level tasks

## 8.1 `utils/reid_batch.py`

Implement or extend:

1. basis-family selector
2. richer row/column basis
3. optional block basis if clean
4. normalized blend-state helper
5. candidate-vs-active logging support
6. clipping diagnostics
7. candidate-guard mode support

## 8.2 `utils/reid_batch_runner.py`

Implement or extend:

1. support for new basis families
2. support for new guard modes
3. richer logging
4. normalized blend-state use
5. observer-update alignment ablation
6. candidate-vs-active distinction in logs
7. automatic summary printing with clipping / fallback information

## 8.3 `systems/polymer/notebook_params.py`

Add new config family or profile for the improved workflow.

Do not remove the current `reid_batch` defaults.

## 8.4 `RL_assisted_MPC_reid_batch_unified.ipynb` or new v2 notebook

Prefer adding a new notebook entrypoint or clearly separated config path such as:

- `RL_assisted_MPC_reid_batch_v2_unified.ipynb`

It should expose the new debug knobs and make ablation runs easy.

## 8.5 plotting

Update plotting utilities so the new diagnostics are visible without hand-editing notebook code.

## 8.6 report

Update or add a report file in `report/` describing:

- why the first implementation underperformed
- the new basis family
- the new safety structure
- the new normalized blend state
- the observer-update alignment ablation
- the comparison plan

## 9. Validation and acceptance tests

The task is not complete unless these checks are possible and demonstrated.

### 9.1 Legacy compatibility

1. The old `reid_batch` workflow still runs.
2. Existing matrix / residual / weights / combined workflows remain unchanged.

### 9.2 Diagnostic ablations

3. `eta = 0`, identification off reproduces nominal MPC.
4. `eta = 0`, identification on proves whether identification itself changes anything.
5. `eta = 1`, identification on shows whether the candidate identified model is intrinsically useful or harmful.
6. learned `eta` can be compared directly to those three baselines.

### 9.3 Saturation checks

7. theta clipping diagnostics are logged and plotted.
8. candidate-vs-active model deltas are logged and plotted.
9. it is easy to verify whether `B` is still saturating.

### 9.4 Observer safety

10. the observer still uses only nominal matrices.
11. if observer-update alignment is changed for the ablation, the observer is still nominal, only the measurement timing alignment changes.

### 9.5 Improvement target

12. The improved workflow should show at least one of the following compared with the current version:

- more informative identified model variation
- reduced clipping / reduced cap saturation
- clearer ablation separation
- improved learned reward trend
- meaningful difference between `eta = 0`, `eta = 1`, and learned `eta`

## 10. Final instruction to Codex

Do not just tweak one number and stop.

Implement this as a structured improvement pass with:

1. a preserved legacy baseline
2. a richer v2 basis
3. less overconstrained candidate selection
4. normalized blend-state diagnostics
5. observer-timing ablation
6. strong ablation and plotting support

The output should make it obvious whether the main bottleneck was:

- weak basis family
- overconstraint and saturation
- poor RL state design
- observer-update alignment
- or some combination of the above

## 11. Deliverables

At minimum, Codex should produce:

1. updated `utils/reid_batch.py`
2. updated `utils/reid_batch_runner.py`
3. updated `systems/polymer/notebook_params.py`
4. updated or new unified notebook for the improved workflow
5. updated plotting support
6. updated report in `report/`
7. short note in `change-reports/`

The code should remain clean, additive, and easy to compare against the current workflow.