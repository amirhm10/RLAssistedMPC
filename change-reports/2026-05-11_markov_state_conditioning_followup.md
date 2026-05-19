# 2026-05-11 Markov State Conditioning Follow-up

## Context

Polymer Markov replay-storage A/B was run with:

- executed-action replay
- requested-action replay

The expectation was that storing requested actions might preserve TD3 authority later in training.

## What we found

- Requested-action replay made TD3 acceptance worse, not better.
- Both replay variants showed severe raw-action saturation: almost every step had `max |a_raw| >= 0.95`.
- Both TD3 and LS proposals were effectively pinned at the current `z_bound = 0.05`.
- The Markov runner was still using a raw concatenated RL state, unlike the conditioned mismatch-state path already used by the matrix and residual workflows.

## Change made

Updated the polymer Markov path to build its RL state from the shared conditioned mismatch-state utilities:

- `resolve_mismatch_settings`
- `make_state_conditioner_from_settings`
- `compute_tracking_scale_now`
- `build_rl_state`
- `get_rl_state_dim`

The Markov state now appends:

- conditioned mismatch base state
- `z_prev`
- safe LS correction
- LS score and LS gain-drift diagnostics

The result bundle now also records:

- `markov_base_state_norm_stats`
- `markov_state_mode = "mismatch_conditioned"`
- `markov_mismatch_feature_transform_mode`

## Expected effect

This does not change the plant, reward, or fallback logic. It changes the actor/critic input geometry so TD3 no longer trains on an unconditioned state whose large raw coordinates dominate the smaller `z` and score features.

If TD3 still loses authority after this fix, the next experiment should target the action-range geometry rather than replay storage.
