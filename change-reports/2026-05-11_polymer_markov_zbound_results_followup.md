# 2026-05-11 Polymer Markov z-bound results follow-up

## Context

Reviewed the first polymer Markov action-range A/B after the state-conditioning fix:

- conditioned reference run with the earlier range
- widened `z_bound = 0.08` run

## What we found

- Widening `z_bound` improved mean reward slightly.
- It did **not** improve late TD3 ownership.
- TD3 accepted fraction fell in both the early post-warm-start window and the last 50 episodes.
- LS score increased strongly under the larger range.
- Late raw-action saturation returned to nearly full saturation.
- The final test episode stayed better than nominal MPC, but was essentially neutral relative to the `z = 0.05` conditioned run.

## Interpretation

The action-range A/B suggests that the remaining bottleneck is not mainly lack of authority at `z = 0.05`.

Instead:

- more authority helps LS more than it helps TD3
- TD3 still does not stay close enough to the useful LS manifold
- the next likely bottleneck is actor-objective / target alignment rather than replay, state scaling, or a slightly too-small `z_bound`

## Artifacts

- report update in `report/polymer_markov_unified_algorithm_and_tuning_start_2026_05_11.md`
- figures under `report/figures/polymer_markov_zbound_ab_20260511/`
