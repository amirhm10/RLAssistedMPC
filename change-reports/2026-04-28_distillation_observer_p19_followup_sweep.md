# 2026-04-28 Distillation Observer p19 Follow-Up Sweep

## Summary

Replaced the broad temporary observer-pole sweep candidate list in `distillation_MPCOffsetFree_observer_pole_sweep_temp.ipynb` with a focused follow-up sweep centered on `p19_uniform_fast`.

## What Changed

- Removed the original `p00` to `p29` exploratory candidate set from the temporary sweep notebook.
- Added eleven temporary `q01` to `q11` candidates:
  - five local uniform neighbors around `p19`
  - four variants that keep the first five poles fixed and vary the last two
  - two variants with a slightly faster front-end and moderate tail poles
- Left the rest of the notebook flow unchanged:
  - nominal mode
  - two episodes
  - fresh Aspen session per candidate
  - per-candidate bundle save and comparison outputs

## Reasoning

The first sweep already showed that slower observers are mostly harmful, while `p19_uniform_fast` is the only clearly usable slower alternative. The next useful experiment is a local sweep around `p19`, not another broad search over qualitatively different pole families.
