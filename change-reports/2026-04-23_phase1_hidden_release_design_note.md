# Phase 1 Hidden Release Design Note

Date: 2026-04-23

## Scope

Extended `report/distillation_warm_start_training_analysis.md` to evaluate a specific Phase 1 proposal:

- keep replay filling during warm start
- freeze the executed action for 5 sub-episodes after warm start
- decide whether critic and/or actor should train during that hidden window

Also updated `report/scripts/distillation_warm_start_analysis.py` to generate:

- `report/figures/distillation_phase1_variant_schedule.png`

## Main conclusion

- The current code already disables both critic and actor learning during replay filling before `warm_start_step`.
- A 5-sub-episode hidden-action window is a good Phase 1 only if actor optimization is frozen during that window.
- Letting actor optimization continue while actions remain frozen is not recommended as the first change, because it can accumulate actor drift before any learned action is deployed.
- Enabling critic learning earlier during warm-start replay fill is plausible, but with the current TD3 target it is not the cleanest first change unless the target actor is anchored to the warm-start action.

## Planning implication

If implemented later, Phase 1 should be modeled as two coordinated controls:

- runner-level executed-action freeze
- agent-level actor optimizer freeze
