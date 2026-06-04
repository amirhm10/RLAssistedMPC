# Distillation SG-SAC Detgate Hidden-7 Transfer

## What Changed

- Added distillation SG-SAC protected-handover wrappers for weights, residual, and Markov correction.
- Updated the distillation weights and residual exported runners to instantiate `SupervisorGatedSACAgent` when `agent_kind="sg_sac"`.
- Updated the distillation Markov exported runner to allow `sg_sac`, pass `sac_agent` into `markov_cfg`, and display SAC/gate settings in the run summary.
- Added SAC agent defaults to the distillation Markov notebook defaults so SG-SAC construction is available from the canonical config.

## Handover Settings

- `warm_start_override = 10`
- `post_warm_start_action_freeze_subepisodes = 10`
- `post_warm_start_actor_freeze_subepisodes = 3`
- `alpha_freeze = "actor_freeze"`
- `n_step = 1`, `multistep_mode = "one_step"`, `actor_q_mode = "min"`

## Gate Settings

- Weights/residual use deterministic candidate scoring, twin-critic dominance, `advantage_margin = 0.5`, and `critic_dominance_margin = 0.5`.
- Markov uses the same deterministic dominance gate with `advantage_margin = 0.0` and `critic_dominance_margin = 0.0`.
- All three wrappers enable the SG-SAC sampled supervisor anchor with `sampled_supervisor_bc_weight = 0.01`.

## Safety Layer Scope

- Old live BC handoff, release gate, tail anchor, authority ramp, and reward probation behavior are disabled for these SG-SAC wrappers.
- Distillation weights keep nonfinite identity fallback and shadow identity diagnostics.
- Distillation residual keeps nonfinite zero fallback and shadow residual diagnostics.
- Distillation Markov disables live z-safety, priority fallback, and authority ramp while retaining shadow Markov safety diagnostics.

## Validation

- Added `tests/test_distillation_supervisor_gated_sac_runners.py` for wrapper configuration and Markov SG-SAC construction smoke coverage.
