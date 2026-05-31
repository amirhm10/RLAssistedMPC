# Distillation Post-Reward No-Probation Analysis

Date: 2026-05-31

## Summary

Added a detailed five-runner analysis for the latest distillation batch after reward probation was disabled and the temperature reward weight was increased to `Q2 = 2.0e4`.

## Files Added

- `report/distillation_post_reward_no_probation_5runner_analysis_2026_05_31.md`
- `report/scripts/analyze_distillation_post_reward_no_probation_20260531.py`

## Main Findings

- Residual TD3 is the clear current leader, with strong tail reward and improved composition and temperature tracking versus OF-MPC.
- Residual still has a severe early release crash, so the next change should be early-release protection only, not active rho.
- Weights improved temperature enough to beat OF-MPC in scalar reward, but it worsened composition and outside-band frequency.
- Standard and dueling horizon runners were not blocked by safety in the tail. Their main issue appears to be poor action-space concentration after widening to 263 horizon pairs.
- Markov is no longer being forced into nominal copy-paste. It is live TD3, but the actor saturates to a fixed Markov action corner. z-safety projects that corner to about `abs(z_i) = 0.03` because the vector norm cap is `0.06`.
- Reward probation is inactive in all five latest bundles.

## Validation

- The analysis script was run against saved bundles only.
- Generated figures were checked for nonblank pixel statistics.
- No Aspen simulation was run.
