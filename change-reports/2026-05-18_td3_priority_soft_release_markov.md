# TD3-Priority Soft Release For Markov Runs

Date: 2026-05-18

## Summary

Added shared TD3-priority soft-release behavior for polymer and distillation Markov unified runs.

## Changes

- Extended polymer and distillation Markov defaults with:
  - TD3 authority ramp from `0.25` to `1.0`
  - reward-collapse probation
  - two-subepisode cooldown at authority scale `0.25`
- Updated the shared Markov runner to:
  - preserve raw actor actions in diagnostics
  - evaluate/execute scaled TD3 requested actions in TD3-priority mode
  - keep `force_td3_execute=True` behavior unchanged
  - keep the old positive-score veto disabled in TD3-priority mode
  - log TD3 phase, authority scale, probation activity, and probation triggers
- Updated Markov diagnostics export and plotting bundle persistence for the new logs.
- Updated polymer and distillation unified Markov notebook setup displays to show soft-release settings before long runs.

## Validation

- Non-Aspen checks only.
- No notebooks were executed.
- No Aspen execution was performed.
