# 2026-05-09 Markov code discrepancy audit

## Objective

Audit the old polymer Markov prototype script and the current shared Markov runner line by line to identify the concrete implementation differences behind the behavior discrepancy.

## Files changed

- `report/polymer_markov_latest_run_analysis_2026_05_09.md`

## Main finding

The largest high-confidence control-loop discrepancy is the disturbance application order:

- the old prototype wrote `Qi/Qs/hA` into `PolymerCSTR` before `system.step()`
- the shared helper `step_system_with_disturbance(...)` currently steps first and then writes dict-style polymer disturbances when no `system_stepper` override is provided

That creates a one-step disturbance lag in the current shared polymer runner, which can materially change the prediction-error windows, LS fit, TD3 state, and accepted corrections even when the rest of the TD3/LS logic is nearly identical.

## Secondary findings

- The TD3/LS gating and replay logic are mostly the same between the two implementations.
- The nominal-solver difference is real but does not appear to be the dominant cause.
- Reward and comparator differences affect the reported win/loss story, but are not the best explanation for why the new live loop stays close to nominal MPC.
