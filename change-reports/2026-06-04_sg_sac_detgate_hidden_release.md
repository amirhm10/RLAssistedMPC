# SG-SAC Deterministic Gate and Hidden Release

## Summary

Implemented the shared SG-SAC deterministic gate update for the polymer
residual, weights, and Markov critic-warm runners. The live gate can now score
the actor mean instead of a stochastic SAC sample, and an optional twin-critic
dominance veto requires both critics to prefer the policy candidate over the
supervisor before release.

## Algorithm Changes

- Added `candidate_mode` to `SACSupervisorGateConfig` with `sampled` and
  `deterministic` modes.
- Added `critic_dominance_gate_enabled` and `critic_dominance_margin`.
- Added sampled-action supervisor anchoring through
  `sampled_supervisor_bc_weight * E[w_sup * ||a_sample - a_supervisor||^2]`.
- Preserved the existing conservative score and advantage-margin gate; critic
  dominance is an additional rejection test when enabled.

## Runner Defaults

- Residual and weights SG-SAC use deterministic candidates, critic dominance
  margin `0.5`, and sampled supervisor BC weight `0.01`.
- Markov SG-SAC uses deterministic candidates, critic dominance margin `0.0`,
  and sampled supervisor BC weight `0.01`.
- All three wrappers now use 10 post-warm-start action-freeze subepisodes and
  3 post-warm-start actor/alpha-freeze subepisodes, giving 7 hidden
  actor/alpha-training subepisodes before live policy execution.
- Result prefixes include `detgate_hidden7` so these runs are distinguishable
  from earlier SG-SAC critic-warm runs.

## Validation Scope

Targeted validation covers deterministic candidate selection, the twin-critic
dominance veto, sampled supervisor BC logging, runner config defaults, and
syntax checks for the edited SG-SAC and runner files. Full 160k-step training is
left as the acceptance experiment.
