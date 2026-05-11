# 2026-05-11 Markov Step 3 Probe: Disturbance Key Mismatch

## Objective

Reproduce the old polymer Markov Step 3 reward trend in a temporary probe surface using a copied notebook and a copied shared runner with `n_tests = 30`.

Target behavior from the historical Step 3 / legacy-like runs:

- average reward should improve after the early episodes
- the 30-episode mean should move near `-4.03` instead of staying near `-4.43`
- nominal fallback fraction should stay near `0.01` instead of `0.03+`

## Files inspected

- `RL_assisted_MPC_markov_unified.ipynb`
- `RL_assisted_MPC_markov_unified_step3_probe.ipynb`
- `utils/markov_runner.py`
- `utils/markov_runner_step3_probe.py`
- `utils/helpers.py`
- `Simulation/system_functions.py`
- historical mimic runner from `7b941bf:utils/markov_runner_legacy_mimic.py`

## Key finding

The consolidated shared runner path was not reproducing Step 3 semantics because the polymer disturbance schedule was being built with lowercase keys:

- `qi`
- `qs`
- `ha`

but `PolymerCSTR` reads plant-side disturbance attributes from:

- `Qi`
- `Qs`
- `hA`

So the helper-based disturbed live-step path silently wrote the wrong attribute names on the plant object.

## Evidence

### Current shared-runner probe before the temporary fix

Run directory:

- `Polymer/Results/td3_markov_disturb_step3_probe/20260511_011627`

Observed:

- `reward_mean = -4.425754109392036`
- `reward_final_episode = -4.453550254894402`
- `nominal_fallback_fraction = 0.03675`
- `prediction_score_mean = 0.024434586120740325`

This remained close to the nominal-like behavior the user reported.

### Historical Step 3 mimic runner

Run directory:

- `Polymer/Results/td3_markov_disturb_step3_probe/20260511_012855`

Observed:

- `reward_mean = -4.029895784495305`
- `reward_final_episode = -3.896936426594957`
- `nominal_fallback_fraction = 0.010833333333333334`
- `prediction_score_mean = 0.0323226933646016`

This matches the desired legacy-like trend.

### Temporary minimal fix in the copied shared runner

Temporary change in `utils/markov_runner_step3_probe.py`:

- emit disturbance schedule keys as `Qi`, `Qs`, `hA` for the polymer disturbed live path

Run directory after that change:

- `Polymer/Results/td3_markov_disturb_step3_probe/20260511_015409`

Observed:

- `reward_mean = -4.029895784495305`
- `reward_final_episode = -3.896936426594957`
- `nominal_fallback_fraction = 0.010833333333333334`
- `prediction_score_mean = 0.0323226933646016`

This exactly matched the old Step 3 probe behavior.

## Conclusion

The discrepancy was not warm start, seed handling, or TD3 construction. The missing functional parity was the polymer disturbance attribute naming in the consolidated helper path.

For polymer Step 3 parity, the disturbed live runner must write:

- `system.Qi`
- `system.Qs`
- `system.hA`

not lowercase `qi/qs/ha`.
