# Restore distillation Markov five-layer safety stack

Date: 2026-05-29

## Summary

- Restored the active distillation Markov default to a TD3-priority safety stack instead of forced TD3 execution.
- Re-enabled LS and nominal emergency fallback, reward probation, priority authority scaling, BC handoff, and dynamic z-safety.
- Added diagnostic-only protected BC release-gate support so release-gate pass/block evidence is logged without blocking live TD3.
- Added a requested-candidate legacy hard-gate pass log for analysis of the old strict score/cost/drift gate.

## Markov default changes

- `force_td3_execute = False`
- `rl_fallback_to_ls = True`
- `td3_priority_fallback.enabled = True`
- `z_bound = 0.04`
- z-safety caps: protected `0.025`, ramp `0.03 -> 0.04`, full `0.04`, probation `0.025`, vector norm cap `0.06`
- Markov BC loss is limited to warm start, while the raw-action handoff remains active for post-warm release.

## Validation

- `py_compile` passed for Markov config/helper/runner files.
- No-Aspen config smoke check passed for restored safety switches.
- Behavioral smoke check passed for diagnostic-only release gate and post-warm handoff timing.
- Markov runner import and Markov BC schedule smoke checks passed.
