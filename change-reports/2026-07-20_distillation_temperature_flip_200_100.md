# Distillation Temperature-Flip 200+100 Schedule

Date: 2026-07-20

## Outcome

- Added the canonical distillation training profile `temperature_flip_200_100`.
- Made the profile the disturbed-fluctuation default for OF-MPC and every active distillation supervisor.
- Restored SG defaults while retaining plain-agent selections:
  - horizon: SG-DQN;
  - Markov, weights, and residual: SG-TD3;
  - combined: all-SG.
- Replaced the temporary 10+10 nominal OF-MPC entrypoint setup with the new 300-episode disturbed baseline.

## Canonical Schedule

Every episode contains 400 samples and two 200-sample setpoint holds.

- Episodes 1--200: `[0.013, -23]`, followed by `[0.028, -21]`.
- Episodes 201--300: `[0.013, -21]`, followed by `[0.028, -23]`.
- Total samples: 120,000.
- Phase boundary: sample 80,000, episode 201.

The first 80,000 feed-disturbance samples are exactly the seed-42 fluctuation sequence used by the legacy 200-episode experiment. Phase 2 retains the same `RandomState` and terminal feed value, then consumes the subsequent random draws with the same fluctuation algorithm. Consequently, the first Phase-2 feed sample equals the final Phase-1 sample.

## Runtime Behavior

- A shared validated `runtime_ctx["episode_bundle"]` now supplies the complete setpoint schedule, episode bookkeeping, named disturbance schedule, phase metadata, and exploration-freeze configuration.
- Polymer behavior remains the fallback when an episode bundle is absent.
- Exploration-schedule amplitude freezes at sample 80,000. Stochastic action sampling and learning continue through episode 299; episode 300 is evaluation-only.
- The matching disturbed baseline uses the new distinct path:
  `Distillation/Data/mpc_results_disturb_fluctuation_temperature_flip_200_100.pickle`.
- RL result and comparison prefixes include the training-profile suffix, and baseline lookup is profile-aware.
- Missing or incompatible baselines no longer prevent saving an RL result; comparison is skipped with a warning.

## Plotting And Saved Metrics

- Generalized the previous polymer-specific two-phase plot path to consume profile metadata for either plant.
- Added phase-boundary plots for outputs/setpoints/inputs, named disturbances, reward/effective exploration, and focused episode views.
- Saved physical tracking RMSE and mean reward for episodes 191--200, 201--210, and 291--300.
- Baseline compatibility now compares the complete setpoint schedule and every named disturbance series instead of assuming polymer-only `Qi`, `Qs`, and `hA` fields.

## Consistency Correction

Restoring SG as the active default exposed an older default mismatch: distillation Markov SG-TD3 had drifted to `param_noise_std_start = 0.10`, despite its established test and the 2026-06-10 ablation report specifying `0.05`. The active default is restored to `0.05`; the residual SG-TD3 default remains `0.10`.

## Verification

- Confirmed exact equality of the new Phase-1 setpoints and all 80,000 feed samples against the legacy generator.
- Confirmed flipped Phase-2 targets, 120,000-sample length, episode-201 boundary, feed continuity, deterministic RNG continuation, exploration-freeze metadata, and final evaluation-only episode.
- Generated synthetic 300-episode plots for baseline, horizon, Markov, weights, residual, and combined families.
- Verified a one-sample setpoint or feed alteration rejects baseline compatibility.
- In-memory syntax compilation passed for all edited Python modules and entrypoints.
- Full automated suite: `103 passed`.
- Aspen was not launched during automated validation.

## Files Changed

- Active distillation entrypoints: baseline, horizon, Markov, weights, residual, and combined.
- Distillation scenario, configuration, defaults, exports, and baseline-path modules.
- Shared episode-profile validation, baseline/supervisor runners, and plotting code.
- Focused distillation profile and combined-runner tests.
