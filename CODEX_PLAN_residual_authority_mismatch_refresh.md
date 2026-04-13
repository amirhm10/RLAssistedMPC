# Codex Plan: Residual Executed-Action, Authority, and Mismatch Refresh

## Goal

Implement a focused residual-only cleanup in the repo.

This pass is **not** a full RL-wide refactor. It is limited to the residual method and any shared helper that the residual method directly depends on.

The three targets are:

1. preserve and verify executed residual action saving
2. rethink and redesign the shared `rho` / authority behavior
3. redesign mismatch-state behavior for the residual method, including adding `rho` to the RL state

## Scope

### In scope

- `utils/residual_runner.py`
- `utils/state_features.py`
- `systems/polymer/notebook_params.py`
- `systems/distillation/notebook_params.py`
- `RL_assisted_MPC_residual_unified.ipynb`
- `distillation_RL_assisted_MPC_residual_unified.ipynb`
- residual-related plotting / result-bundle normalization if needed
- the residual block inside `utils/combined_runner.py` and the combined notebooks **only if** they call the same residual authority or mismatch-state helpers and would otherwise drift from the residual notebooks

### Out of scope for this pass

- horizon method redesign
- matrix method redesign
- weight method redesign
- replay-buffer redesign outside the residual method
- generic TD3 / SAC architecture changes unless a very small residual-only wrapper is required

## High-level design decisions

### Decision 1: executed residual action remains the learning action stored in replay

The buffer must continue to store the **executed** residual action after all clipping / authority projection, not the raw actor proposal.

Keep the current intent:

- actor outputs raw normalized action
- raw action maps to raw residual move in scaled deviation units
- residual move is projected by authority and actuator headroom
- projected / executed residual is mapped back to normalized action space
- replay stores the normalized executed residual action

This must be audited and made explicit in code comments and logs.

### Decision 2: centralize the residual projection logic

Create one shared helper for the residual method that computes the executed residual action.

That helper should accept the current rollout quantities and return a structured object with at least:

- `delta_u_res_raw`
- `delta_u_res_exec`
- `a_raw`
- `a_exec`
- `rho`
- authority bounds before actuator headroom
- final lower / upper bounds after actuator headroom
- flags for which limiter was active

This helper should be used everywhere the residual action is formed so the logic cannot drift.

### Decision 3: redesign mismatch-state behavior for residual mode

Do **not** keep the current single `mismatch_scale` built from setpoint range.

Replace it with separate handling for:

- innovation feature
- tracking-error feature
- `rho` feature

Use the following design:

#### Innovation feature

Use a fixed per-output reference scale based on reward bands, not setpoint span.

Recommended default:

- `innovation_scale_ref = band_ref_scaled`

where `band_ref_scaled` is the median scaled reward band across the residual notebook RL setpoints.

Then compute:

```text
innovation_feat = clip((y_prev_scaled - yhat_pred) / innovation_scale_ref, -mismatch_clip, mismatch_clip)
```

#### Tracking-error feature

Normalize tracking error by the same tolerance structure used by authority.

Recommended default:

```text
tracking_scale = max(eta_tol * band_scaled_now, tracking_scale_floor)
tracking_error_feat = clip((y_prev_scaled - y_sp) / tracking_scale, -mismatch_clip, mismatch_clip)
```

where `tracking_scale_floor` is a per-output floor derived from the reference band, for example:

```text
tracking_scale_floor = 0.5 * eta_tol * band_ref_scaled
```

#### Rho feature

Append `rho` directly to the RL state.

Do **not** force the policy to infer authority only from weak mismatch channels.

### Decision 4: redesign `rho` / authority so it stays conservative but does not collapse too aggressively

The current idea is useful but too restrictive near the setpoint.

Keep the idea that larger normalized tracking error gives more residual authority, but make it configurable and easier to inspect.

Recommended structure:

```text
track_norm_i = abs(tracking_error_feat_i)
rho_raw = max_i(track_norm_i)
rho = clip(rho_raw, 0, 1)
```

Then allow a configurable floor:

```text
rho_eff = rho_floor + (1 - rho_floor) * rho
```

with default `rho_floor > 0` so residual authority never fully disappears.

Use `rho_eff` instead of bare `rho` when forming authority limits.

### Decision 5: expose authority and mismatch parameters in notebook defaults

Do not keep these as hidden constants in `utils/residual_runner.py`.

Add them to both polymer and distillation residual defaults.

Recommended residual config block:

- `authority_use_rho`
- `authority_beta_res`
- `authority_du0_res`
- `authority_eta_tol`
- `authority_rho_floor`
- `authority_rho_power` or keep linear if not needed
- `mismatch_clip`
- `innovation_scale_mode`
- `innovation_scale_ref`
- `tracking_scale_mode`
- `tracking_scale_floor`
- `append_rho_to_state`

## Detailed implementation tasks

### Task A: audit and preserve executed-action saving

Files:

- `utils/residual_runner.py`
- any residual path inside `utils/combined_runner.py`

Required work:

1. Find the exact point where raw residual action becomes executed residual action.
2. Make that path explicit and well named.
3. Confirm replay always stores `a_exec`, not `a_raw`.
4. Add comments explaining why this is required.
5. Add result-bundle logs for both raw and executed actions.

Required logs:

- `a_res_raw_log`
- `a_res_exec_log`
- `delta_u_res_raw_log`
- `delta_u_res_exec_log`
- `projection_active_log`
- `projection_due_to_authority_log`
- `projection_due_to_headroom_log`

Acceptance check:

- if projection is inactive, raw and executed actions match exactly
- if projection is active, replay stores the executed action

### Task B: create one shared residual projection helper

Preferred location:

- either inside `utils/residual_runner.py` as a well-contained helper
- or a small new helper module such as `utils/residual_authority.py`

The helper should:

1. map normalized actor action to raw residual move
2. compute current band quantities needed for authority
3. compute tracking-based normalized quantities and `rho`
4. compute authority magnitude bounds
5. apply actuator headroom around `u_base`
6. clip to final executed residual move
7. map executed move back to normalized action space
8. return all intermediate diagnostics

This helper must be the single source of truth for residual projection.

### Task C: redesign mismatch state builder for residual mode

Files:

- `utils/state_features.py`
- residual notebooks
- combined residual path if needed

Change the residual mismatch-state API from one shared `mismatch_scale` to separate fields.

Recommended new API shape:

- `innovation_scale_ref`
- `tracking_scale_now`
- `append_rho`
- `rho_value`
- `mismatch_clip`

Implement a residual-mode state builder that returns:

```text
base_state
innovation features
tracking-error features
rho feature (optional, default True for residual)
```

Important:

- keep backward compatibility for non-residual methods if they still call the old mismatch path
- do not break horizon / matrix / weights notebooks in this pass
- if necessary, add a residual-specific state mode such as `mismatch_residual` rather than forcing a repo-wide breaking change

### Task D: compute reference bands from existing reward settings

Files:

- residual notebooks or a small helper function used by them

Use the existing reward configuration and RL setpoints to build a reference band per output.

For each residual notebook:

1. read current reward config
2. compute `band_scaled` for each RL setpoint
3. take the median across those setpoints
4. store as `band_ref_scaled`

Use that as the default `innovation_scale_ref`.

Do not hard-code numeric values into shared logic unless needed for fallback.

### Task E: redesign authority parameters and move them to notebook defaults

Files:

- `systems/polymer/notebook_params.py`
- `systems/distillation/notebook_params.py`
- residual notebooks

Add a dedicated residual authority config block.

Suggested starting defaults:

- `authority_use_rho = True`
- `authority_rho_floor = 0.15`
- `authority_eta_tol = 0.3`
- `authority_beta_res = existing values, but exposed`
- `authority_du0_res = existing values, but exposed`
- `append_rho_to_state = True`
- `mismatch_clip = 3.0`
- `innovation_scale_mode = "band_ref"`
- `tracking_scale_mode = "eta_band"`
- `tracking_scale_floor_mode = "half_eta_band_ref"`

Do not silently change the old defaults without documenting them.

### Task F: update notebooks to pass the new config cleanly

Files:

- `RL_assisted_MPC_residual_unified.ipynb`
- `distillation_RL_assisted_MPC_residual_unified.ipynb`
- combined notebooks only if residual branch uses the same shared logic

Required notebook changes:

1. surface the new residual authority and mismatch config in the top config cell
2. pass these into `residual_cfg`
3. include them in the run summary printout
4. ensure result-prefix naming remains distinct when relevant

### Task G: enrich result bundles and plotting for diagnosis

Files:

- residual plotting layer and normalization helpers if needed

Add plots or stored arrays for:

- `rho_log`
- `rho_eff_log`
- fraction of steps where projection is active
- fraction of steps limited by authority
- fraction of steps limited by actuator headroom
- per-input mean absolute raw vs executed residual
- histograms or traces of innovation and tracking-error features if those diagnostics already fit the plotting style

This is needed so future runs can answer:

- is the actor weak?
- or is the authority shrinking the residual channel too much?

## Recommended exact behavior for the first implementation

Use this as the first-pass default behavior.

### Residual authority

```text
track_norm = abs(tracking_error_feat)
rho_raw = max(track_norm)
rho = clip(rho_raw, 0, 1)
rho_eff = rho_floor + (1 - rho_floor) * rho
mag = (rho_eff * beta_res) * (abs(delta_u_mpc) + du0_res)
```

Keep `beta_res` and `du0_res` vector-valued per input.

Do not remove actuator headroom clipping.

### Innovation feature

```text
innovation_scale_ref = band_ref_scaled
innovation_feat = clip((y_prev_scaled - yhat_pred) / innovation_scale_ref, -mismatch_clip, mismatch_clip)
```

### Tracking-error feature

```text
tracking_scale_now = max(eta_tol * band_scaled_now, tracking_scale_floor)
tracking_error_feat = clip((y_prev_scaled - y_sp) / tracking_scale_now, -mismatch_clip, mismatch_clip)
```

### State content for residual mismatch mode

```text
state = [base_state, innovation_feat, tracking_error_feat, rho]
```

If `rho` is scalar, append one extra dimension.

## Important non-goals for this pass

Do not do these unless absolutely necessary:

- large TD3 / SAC internal refactors
- changing generic replay buffer behavior for all methods
- changing the reward function itself
- changing non-residual notebook defaults outside compatibility fixes

## Validation checklist

### Code-level validation

- all touched Python modules import and compile
- notebooks still load and resolve configuration cleanly
- no shape mismatch in residual state dimension
- no breakage in standard state mode

### Behavioral validation

Run shortened smoke tests for:

- polymer residual notebook, nominal
- polymer residual notebook, disturb
- distillation residual notebook, nominal
- distillation residual notebook, disturb

For each smoke test, verify:

1. replay stores executed residual action
2. `rho_log` and `rho_eff_log` are populated when enabled
3. mismatch-state dimensions match the new design
4. residual projection helper is actually used
5. action logs show nontrivial difference between raw and executed actions on at least some steps
6. residual authority does not fully collapse near the setpoint because of the new floor

### Diagnostics to inspect manually

- average `rho`
- average `rho_eff`
- clipping fraction
- mean absolute raw residual vs executed residual
- average absolute innovation feature by output
- average absolute tracking-error feature by output

## Deliverables

Codex should produce:

1. code changes for the residual-only refresh
2. updated residual notebook config surfaces
3. richer result-bundle logging for residual runs
4. a short change report markdown file describing exactly what changed and why

Suggested change report path:

- `change-reports/YYYY-MM-DD_residual_authority_mismatch_refresh.md`

## Notes for Codex

- Keep edits targeted and minimal.
- Do not drift into horizon / matrix / weight redesign.
- Preserve current behavior where it is clearly intentional.
- When making shared-helper changes, keep backward compatibility unless there is a strong reason not to.
- Prefer small pure functions for authority and mismatch-state transformations so the behavior is easy to test and easy to review.
