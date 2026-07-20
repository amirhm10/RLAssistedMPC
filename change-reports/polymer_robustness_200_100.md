# Polymer robustness_200_100 implementation

## Outcome

The default disturbed polymer workflow now uses a 300-episode, two-phase robustness profile shared by the offset-free MPC baseline and the horizon, Markov, weight, residual, and combined RL supervisors. The previous 200-episode generator remains available as `legacy_gradual_200`.

Setting only `training_profile_override = "legacy_gradual_200"` restores the associated 200-episode count unless an explicit `n_tests_override` is also supplied.

The new baseline is stored separately as:

`Polymer/Data/mpc_results_dist_robustness_200_100.pickle`

The existing `Polymer/Data/mpc_results_dist.pickle` path is unchanged for the legacy profile.

## Schedule

Each episode has two 400-step holds, so the phase boundary is step 160,000 and the total run is 240,000 steps.

- Episodes 1–200 reuse the legacy step-level setpoint and disturbance generator exactly.
- Episodes 201–300 use physical setpoints `[4.0, 321.5]` and `[3.3, 324.5]`.
- For Phase 2 step index `j = 0, ..., 79,999`:

$$ Q_i(j) = 91.8 + (102.6 - 91.8)\frac{j}{79,999}. $$

$$ Q_s(j) = 596.7 + (481.95 - 596.7)\frac{j}{79,999}. $$

$$ hA(j) = 892,500. $$

The first Phase-2 disturbance sample equals the final Phase-1 sample for `Qi` and `Qs`, so there is no reset or discontinuity at the boundary. The fouling coefficient remains at its Phase-1 terminal value.

## Exploration and learning

At environment step 160,000, every active agent stores its current exploration-schedule amplitude. DQN epsilon and TD3 Gaussian/parameter-noise schedules then use that stored amplitude through episode 299. Random noise continues to be drawn normally; only the amplitude annealing is frozen.

Episode 300 passes the existing evaluation flag to the action and replay/training paths, which makes the recorded effective exploration amplitude zero and prevents replay/training updates. Episodes 201–299 remain training episodes. Learning rates, replay-priority annealing, target-network updates, optimizer state, replay buffers, observers, plant state, and MPC state are not reset or frozen.

## Defaults and provenance

- Horizon: `sg_dqn`
- Markov, weights, residual: `sg_td3`
- Combined: all-SG mode
- Training profile: `robustness_200_100`
- Result and comparison prefixes: suffixed with `_robustness_200_100`

Saved bundles include phase metadata, exploration-freeze settings and values, step-level effective exploration traces, persistent-fouling metadata, and Phase-2 learning/evaluation status. Baseline comparisons verify the profile label, dimensions, phase boundary, complete setpoint schedule, and complete `Qi/Qs/hA` schedules. A missing or incompatible robustness baseline produces a warning and skips comparison after the RL run has been saved.

## Plotting and metrics

The shared plotting layer adds robustness-study figures with an episode-201 marker for:

- outputs, setpoints, and inputs;
- `Qi`, `Qs`, and `hA`;
- average episode reward and effective exploration amplitude;
- focused episode-201 and episode-300 output and input views.

Saved result bundles also contain physical tracking RMSE per output and average reward for episodes 191–200, 201–210, and 291–300.

## Verification

- The first 160,000 setpoint, `Qi`, `Qs`, and `hA` samples were checked for exact equality with the legacy generator.
- Phase-2 setpoints, endpoints, continuity, and constant `hA = 892,500` were checked directly.
- DQN epsilon and both TD3 exploration modes were checked for a constant post-boundary schedule value and zero evaluation value.
- Polymer SG defaults, distinct baseline paths, legacy-profile availability, missing-baseline behavior, and schedule-mismatch rejection were checked.
- Synthetic 300-episode plots were generated and visually inspected.
- No full 240,000-step plant/controller experiment was run during validation.

## Remaining experimental uncertainty

The implementation verifies schedule and control-flow correctness, not closed-loop robustness performance. Tracking quality, constraint activity, supervisor selection, replay adaptation, and learning stability in episodes 201–300 must be assessed from the first real SG runs and their saved window metrics.

## 2026-07-20 default-mode transition

After the SG robustness runs were completed, the active polymer defaults were changed to the matching plain-agent arm: DQN for horizon, TD3 for Markov/weights/residual, and all-plain combined mode. The `robustness_200_100` schedule, matching MPC baseline, disturbance profile, exploration-freeze behavior, and result naming remain unchanged. SG configurations and their existing artifacts remain selectable and are not modified.
