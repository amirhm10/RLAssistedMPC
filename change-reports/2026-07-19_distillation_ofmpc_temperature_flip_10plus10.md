# Distillation OF-MPC Temperature-Flip Experiment

## Objective

Configure one continuous nominal distillation offset-free MPC run with two phases:

- Episodes 1 through 10 use the canonical setpoint mapping.
- Episodes 11 through 20 retain the same composition sequence and exchange the two temperature targets.

The Aspen plant and offset-free observer are not reset at the phase boundary.

## Setpoint schedule

Each episode contains two 200-step target blocks.

| Phase | First target | Second target |
|---|---|---|
| Episodes 1-10 | composition 0.013, temperature -23 | composition 0.028, temperature -21 |
| Episodes 11-20 | composition 0.013, temperature -21 | composition 0.028, temperature -23 |

The physical schedule is transformed with the same min-max and steady-state-deviation mapping used by the existing MPC baseline.

## Implementation

- `utils/mpc_baseline_runner.py` now accepts a validated optional step-by-step setpoint schedule override.
- `distillation_MPCOffsetFree_unified.py` builds the 20-episode physical schedule and passes its scaled-deviation form to the runner.
- The result bundle records the episode targets, phase boundary, and experiment name.
- The pickle uses a timestamped experiment-specific filename under `Distillation/Data` and is opened in exclusive-create mode. Existing canonical baseline pickles cannot be overwritten by this experiment.

## Validation scope

Static checks verify the schedule dimensions, target ordering, scaling round trip, and override validation. Aspen is not launched for smoke validation.
