# Distillation Column Transfer Plan

## Purpose

This document records how the distillation project under:

- `[DIstillation Column Case/RL_assisted_MPC_DL](c:\Users\HAMEDI\Desktop\RL_assisted_MPC_polymer\DIstillation Column Case\RL_assisted_MPC_DL)`

should be transferred into the main repo.

The target is not a second parallel framework. The target is one shared RL-assisted MPC architecture with:

- shared runners
- shared plotting
- shared reward/state/observer logic
- a distillation-specific plant/data/scenario/label layer

This document is planning only. No distillation code has been migrated yet.

## Executive Summary

The distillation case now looks consistent enough to migrate onto the unified architecture already built for the polymer case.

The correct migration target includes:

- unified baseline MPC
- unified horizon notebook
- unified matrix-multiplier notebook
- unified weight-multiplier notebook
- unified residual notebook
- unified combined notebook

The algorithm layer is reusable. The plant layer is not.

That means the right design is:

1. keep the main repo unified modules as the algorithm source of truth
2. add a distillation-specific system adapter
3. add a distillation-specific data/scenario/label layer
4. build distillation unified notebooks on top of those shared modules

Residual should now be included in scope even though the old distillation subproject does not have a mature residual notebook family. The unified residual modules in the main repo are already generic enough to port.

## What Exists In The Distillation Subproject

Main folders:

- `[BasicFunctions](c:\Users\HAMEDI\Desktop\RL_assisted_MPC_polymer\DIstillation Column Case\RL_assisted_MPC_DL\BasicFunctions)`
- `[DQN](c:\Users\HAMEDI\Desktop\RL_assisted_MPC_polymer\DIstillation Column Case\RL_assisted_MPC_DL\DQN)`
- `[SACAgent](c:\Users\HAMEDI\Desktop\RL_assisted_MPC_polymer\DIstillation Column Case\RL_assisted_MPC_DL\SACAgent)`
- `[Simulation](c:\Users\HAMEDI\Desktop\RL_assisted_MPC_polymer\DIstillation Column Case\RL_assisted_MPC_DL\Simulation)`
- `[TD3Agent](c:\Users\HAMEDI\Desktop\RL_assisted_MPC_polymer\DIstillation Column Case\RL_assisted_MPC_DL\TD3Agent)`
- `[utils](c:\Users\HAMEDI\Desktop\RL_assisted_MPC_polymer\DIstillation Column Case\RL_assisted_MPC_DL\utils)`
- `[Data](c:\Users\HAMEDI\Desktop\RL_assisted_MPC_polymer\DIstillation Column Case\RL_assisted_MPC_DL\Data)`
- `[Result](c:\Users\HAMEDI\Desktop\RL_assisted_MPC_polymer\DIstillation Column Case\RL_assisted_MPC_DL\Result)`

Notebook families found:

### Baseline MPC

- `[MPCOffsetFree.ipynb](c:\Users\HAMEDI\Desktop\RL_assisted_MPC_polymer\DIstillation Column Case\RL_assisted_MPC_DL\MPCOffsetFree.ipynb)`
- `[MPCOffsetFreeDistRamp.ipynb](c:\Users\HAMEDI\Desktop\RL_assisted_MPC_polymer\DIstillation Column Case\RL_assisted_MPC_DL\MPCOffsetFreeDistRamp.ipynb)`
- `[MPCOffsetFreeDistRampNoise.ipynb](c:\Users\HAMEDI\Desktop\RL_assisted_MPC_polymer\DIstillation Column Case\RL_assisted_MPC_DL\MPCOffsetFreeDistRampNoise.ipynb)`

### Horizon

- `[RL_assisted_MPC_horizons.ipynb](c:\Users\HAMEDI\Desktop\RL_assisted_MPC_polymer\DIstillation Column Case\RL_assisted_MPC_DL\RL_assisted_MPC_horizons.ipynb)`
- `[RL_assisted_MPC_horizons_ramp.ipynb](c:\Users\HAMEDI\Desktop\RL_assisted_MPC_polymer\DIstillation Column Case\RL_assisted_MPC_DL\RL_assisted_MPC_horizons_ramp.ipynb)`
- `[RL_assisted_MPC_horizons_fluctuation.ipynb](c:\Users\HAMEDI\Desktop\RL_assisted_MPC_polymer\DIstillation Column Case\RL_assisted_MPC_DL\RL_assisted_MPC_horizons_fluctuation.ipynb)`

### Matrices

- `[RL_assisted_MPC_matrices.ipynb](c:\Users\HAMEDI\Desktop\RL_assisted_MPC_polymer\DIstillation Column Case\RL_assisted_MPC_DL\RL_assisted_MPC_matrices.ipynb)`
- `[RL_assisted_MPC_matrices_ramp.ipynb](c:\Users\HAMEDI\Desktop\RL_assisted_MPC_polymer\DIstillation Column Case\RL_assisted_MPC_DL\RL_assisted_MPC_matrices_ramp.ipynb)`
- `[RL_assisted_MPC_matrices_fluctuation.ipynb](c:\Users\HAMEDI\Desktop\RL_assisted_MPC_polymer\DIstillation Column Case\RL_assisted_MPC_DL\RL_assisted_MPC_matrices_fluctuation.ipynb)`
- `[RL_assisted_MPC_matrices_SAC.ipynb](c:\Users\HAMEDI\Desktop\RL_assisted_MPC_polymer\DIstillation Column Case\RL_assisted_MPC_DL\RL_assisted_MPC_matrices_SAC.ipynb)`
- `[RL_assisted_MPC_matrices_ramp_SAC.ipynb](c:\Users\HAMEDI\Desktop\RL_assisted_MPC_polymer\DIstillation Column Case\RL_assisted_MPC_DL\RL_assisted_MPC_matrices_ramp_SAC.ipynb)`
- `[RL_assisted_MPC_matrices_fluctuation_SAC.ipynb](c:\Users\HAMEDI\Desktop\RL_assisted_MPC_polymer\DIstillation Column Case\RL_assisted_MPC_DL\RL_assisted_MPC_matrices_fluctuation_SAC.ipynb)`
- mismatch variants:
  - `[RL_assisted_MPC_matrices_mistmatch.ipynb](c:\Users\HAMEDI\Desktop\RL_assisted_MPC_polymer\DIstillation Column Case\RL_assisted_MPC_DL\RL_assisted_MPC_matrices_mistmatch.ipynb)`
  - `[RL_assisted_MPC_matrices_model_mismatch.ipynb](c:\Users\HAMEDI\Desktop\RL_assisted_MPC_polymer\DIstillation Column Case\RL_assisted_MPC_DL\RL_assisted_MPC_matrices_model_mismatch.ipynb)`
  - `[RL_assisted_MPC_matrices_SAC_mistmatch.ipynb](c:\Users\HAMEDI\Desktop\RL_assisted_MPC_polymer\DIstillation Column Case\RL_assisted_MPC_DL\RL_assisted_MPC_matrices_SAC_mistmatch.ipynb)`

### Weights

- `[RL_assisted_MPC_weights.ipynb](c:\Users\HAMEDI\Desktop\RL_assisted_MPC_polymer\DIstillation Column Case\RL_assisted_MPC_DL\RL_assisted_MPC_weights.ipynb)`
- `[RL_assisted_MPC_weights_ramp.ipynb](c:\Users\HAMEDI\Desktop\RL_assisted_MPC_polymer\DIstillation Column Case\RL_assisted_MPC_DL\RL_assisted_MPC_weights_ramp.ipynb)`
- `[RL_assisted_MPC_weights_fluctuation.ipynb](c:\Users\HAMEDI\Desktop\RL_assisted_MPC_polymer\DIstillation Column Case\RL_assisted_MPC_DL\RL_assisted_MPC_weights_fluctuation.ipynb)`

### Combined

- `[RL_assisted_MPC_combined.ipynb](c:\Users\HAMEDI\Desktop\RL_assisted_MPC_polymer\DIstillation Column Case\RL_assisted_MPC_DL\RL_assisted_MPC_combined.ipynb)`

### System Identification

- `[systemIdentification.ipynb](c:\Users\HAMEDI\Desktop\RL_assisted_MPC_polymer\DIstillation Column Case\RL_assisted_MPC_DL\systemIdentification.ipynb)`

## Plant Runtime

The real plant runtime is Aspen-based through:

- `[Simulation/system_functions.py](c:\Users\HAMEDI\Desktop\RL_assisted_MPC_polymer\DIstillation Column Case\RL_assisted_MPC_DL\Simulation\system_functions.py)`

Observed active class:

- `DistillationColumnAspen`

Observed properties:

- Aspen Dynamics / COM based:
  - `win32com.client.DispatchEx("AD application")`
- manipulated inputs:
  - reflux flow
  - reboiler duty
- outputs:
  - stage-24 ethane composition
  - stage-85 temperature
- feed disturbance injected through the feed stream interface

This confirms the core difference from polymer: the algorithms transfer, the plant setup does not.

## Recheck Result: Updated Source Notebooks

I rechecked the notebooks you said you updated.

### Updated System Identification Notebook

The updated:

- `[systemIdentification.ipynb](c:\Users\HAMEDI\Desktop\RL_assisted_MPC_polymer\DIstillation Column Case\RL_assisted_MPC_DL\systemIdentification.ipynb)`

is now clearly distillation-specific.

Observed indicators:

- `DistillationColumnAspen`
- `run_dl_experiment`
- `Reflux.csv`
- `Reboiler.csv`
- tray-24 ethane composition
- tray-85 temperature

Observed outputs written by the notebook:

- `system_dict_new.pickle`
- `scaling_factor_new.pickle`

So the system-ID notebook is no longer a stale polymer/CSTR notebook. It can now be treated as a valid distillation source notebook.

### Updated Baseline MPC Notebooks

The baseline MPC notebooks are also now distillation-specific:

- `[MPCOffsetFree.ipynb](c:\Users\HAMEDI\Desktop\RL_assisted_MPC_polymer\DIstillation Column Case\RL_assisted_MPC_DL\MPCOffsetFree.ipynb)`
- `[MPCOffsetFreeDistRamp.ipynb](c:\Users\HAMEDI\Desktop\RL_assisted_MPC_polymer\DIstillation Column Case\RL_assisted_MPC_DL\MPCOffsetFreeDistRamp.ipynb)`
- `[MPCOffsetFreeDistRampNoise.ipynb](c:\Users\HAMEDI\Desktop\RL_assisted_MPC_polymer\DIstillation Column Case\RL_assisted_MPC_DL\MPCOffsetFreeDistRampNoise.ipynb)`

Observed save outputs:

- `MPCOffsetFree.ipynb` saves `mpc_results.pickle`
- `MPCOffsetFreeDistRamp.ipynb` saves `mpc_results_dist_ramp.pickle`
- `MPCOffsetFreeDistRampNoise.ipynb` saves `mpc_results_dist_ramp_noise_200.pickle`

So the earlier warning that baseline MPC notebooks were still polymer-based is no longer true.

## Current Distillation Data And Naming State

Important files currently present under:

- `[Data](c:\Users\HAMEDI\Desktop\RL_assisted_MPC_polymer\DIstillation Column Case\RL_assisted_MPC_DL\Data)`

Observed:

- `system_dict.pickle`
- `system_dict_new.pickle`
- `scaling_factor.pickle`
- `scaling_factor_new.pickle`
- `min_max_states.pickle`
- `mpc_results.pickle`
- `mpc_results_200.pickle`
- `mpc_results_dist_ramp.pickle`
- `mpc_results_dist_ramp1.pickle`
- `mpc_result_dist_ramp_noise.pickle`
- `mpc_results_dist_ramp_noise_200.pickle`
- `mpc_results_RL_compare.pickle`

Observed shapes from existing files:

- `A`: `(5, 5)`
- `B`: `(5, 2)`
- `C`: `(2, 5)`
- `D`: `(2, 2)`
- scaling min/max arrays: `(4,)`
- state min/max arrays: `(7,)`

## Distillation Plot Labels

The distillation plotting file:

- `[utils/plotting.py](c:\Users\HAMEDI\Desktop\RL_assisted_MPC_polymer\DIstillation Column Case\RL_assisted_MPC_DL\utils\plotting.py)`

already uses distillation-specific labels.

Observed output labels:

- `x_{24,C2H6} (-)`
- `T_85 (K)`

Observed input labels:

- `Reflux (kg/h)`
- `Reboiler duty (GJ/h)`

Observed time label:

- `Time (h)`

This confirms the right transfer rule: plots stay shared, labels become system metadata.

## Disturbance Families

The distillation notebooks currently separate scenarios into:

- nominal
- ramp disturbance
- fluctuation disturbance
- noisy-ramp / fluctuation-like baseline variant

The polymer notebooks currently use:

- `RUN_MODE = nominal | disturb`

Recommended cross-system abstraction:

- keep `RUN_MODE = nominal | disturb`
- add `DISTURBANCE_PROFILE = none | ramp | fluctuation`

That keeps the control flow aligned with polymer while still reflecting the real distillation scenario families.

## What The RL Distillation Notebooks Currently Compare Against

Observed current compare paths:

### Horizon

- nominal compares to `mpc_results_200.pickle`
- ramp compares to `mpc_results_dist_ramp.pickle`
- fluctuation compares to `mpc_result_dist_ramp_noise.pickle`

### Matrices

- nominal compares to `mpc_results_200.pickle`
- ramp compares to `mpc_results_dist_ramp.pickle`
- fluctuation compares to `mpc_result_dist_ramp_noise.pickle`

### Weights

- nominal compares to `mpc_results_200.pickle`
- ramp compares to `mpc_results_dist_ramp.pickle`
- fluctuation compares to `mpc_result_dist_ramp_noise.pickle`

This is enough evidence that the current scenario semantics are stable, but the file naming is not.

## Remaining Inconsistencies

The remaining migration problems are now mainly naming and architecture consistency.

### 1. Linear-model naming drift

The updated system-ID notebook writes:

- `system_dict_new.pickle`
- `scaling_factor_new.pickle`

but the old names still also exist.

Recommendation:

- migrate to one canonical distillation name in the main repo
- remove the `_new` suffix during transfer

### 2. Baseline MPC naming drift

Current baseline and compare names are inconsistent:

- `mpc_results.pickle`
- `mpc_results_200.pickle`
- `mpc_results_dist_ramp.pickle`
- `mpc_result_dist_ramp_noise.pickle`
- `mpc_results_dist_ramp_noise_200.pickle`

Recommendation:

- normalize these in the main repo

### 3. No native residual distillation source family

I still did not find any distillation residual notebook family.

That is no longer a scope limiter. Residual should still be migrated, but the source should be the main repo’s shared residual implementation, not a distillation source notebook tree.

### 4. Combined notebook coverage is narrower than the current main-repo combined design

The current distillation combined notebook appears to cover:

- horizon DQN
- matrix TD3

It does not appear to already match the polymer-era 4-agent combined architecture. So distillation combined migration should target the current shared combined framework, not preserve the older narrower combined runtime.

## Recommended Main-Repo Architecture

Recommended new structure:

- `[systems/distillation](c:\Users\HAMEDI\Desktop\RL_assisted_MPC_polymer\systems\distillation)`
  - plant adapter
  - scenario definitions
  - data loading
  - label metadata
  - system-level config
- `[Data/distillation](c:\Users\HAMEDI\Desktop\RL_assisted_MPC_polymer\Data\distillation)`
- `[Result/distillation](c:\Users\HAMEDI\Desktop\RL_assisted_MPC_polymer\Result\distillation)`

This is preferable to keeping a second parallel project tree.

## Distillation System Adapter Requirements

The distillation adapter should own the system-specific parts only:

- Aspen init/bootstrap
- steady-state operating point
- nominal conditions
- disturbance injection API
- input/output labels and units
- scenario catalog
- baseline result path mapping
- data-file path mapping

The unified runners should stay shared.

## Recommended Canonical Naming In The Main Repo

### System/model data

- `[Data/distillation/system_dict.pickle](c:\Users\HAMEDI\Desktop\RL_assisted_MPC_polymer\Data\distillation\system_dict.pickle)`
- `[Data/distillation/scaling_factor.pickle](c:\Users\HAMEDI\Desktop\RL_assisted_MPC_polymer\Data\distillation\scaling_factor.pickle)`
- `[Data/distillation/min_max_states.pickle](c:\Users\HAMEDI\Desktop\RL_assisted_MPC_polymer\Data\distillation\min_max_states.pickle)`

### Baseline MPC results

- `[Data/distillation/mpc_results_nominal.pickle](c:\Users\HAMEDI\Desktop\RL_assisted_MPC_polymer\Data\distillation\mpc_results_nominal.pickle)`
- `[Data/distillation/mpc_results_ramp.pickle](c:\Users\HAMEDI\Desktop\RL_assisted_MPC_polymer\Data\distillation\mpc_results_ramp.pickle)`
- `[Data/distillation/mpc_results_fluctuation.pickle](c:\Users\HAMEDI\Desktop\RL_assisted_MPC_polymer\Data\distillation\mpc_results_fluctuation.pickle)`

### Run prefixes

- `distillation_horizon_nominal`
- `distillation_horizon_ramp`
- `distillation_horizon_fluctuation`
- `distillation_matrix_td3_nominal`
- `distillation_matrix_td3_ramp`
- `distillation_matrix_td3_fluctuation`
- `distillation_matrix_sac_nominal`
- `distillation_matrix_sac_ramp`
- `distillation_matrix_sac_fluctuation`
- `distillation_weight_td3_nominal`
- `distillation_weight_td3_ramp`
- `distillation_weight_td3_fluctuation`
- `distillation_weight_sac_nominal`
- `distillation_weight_sac_ramp`
- `distillation_weight_sac_fluctuation`
- `distillation_residual_td3_nominal`
- `distillation_residual_td3_ramp`
- `distillation_residual_td3_fluctuation`
- `distillation_residual_sac_nominal`
- `distillation_residual_sac_ramp`
- `distillation_residual_sac_fluctuation`
- `distillation_combined_nominal`
- `distillation_combined_ramp`
- `distillation_combined_fluctuation`

## Recommended Distillation Unified Notebook Surface

Recommended notebook files:

- `distillation_MPCOffsetFree_unified.ipynb`
- `distillation_RL_assisted_MPC_horizons_unified.ipynb`
- `distillation_RL_assisted_MPC_matrices_unified.ipynb`
- `distillation_RL_assisted_MPC_weights_unified.ipynb`
- `distillation_RL_assisted_MPC_residual_unified.ipynb`
- `distillation_RL_assisted_MPC_combined_unified.ipynb`

Optional later:

- `distillation_systemIdentification_unified.ipynb`

Why separate distillation notebooks instead of one notebook with `SYSTEM = polymer | distillation`:

- Aspen runtime dependencies are different
- scenario catalogs are different
- labels and units are different
- baseline path mappings are different
- the notebook should already open in the correct plant context

## Mapping To Existing Main-Repo Shared Modules

### Already reusable

- `[utils/rewards.py](c:\Users\HAMEDI\Desktop\RL_assisted_MPC_polymer\utils\rewards.py)`
- `[utils/observer.py](c:\Users\HAMEDI\Desktop\RL_assisted_MPC_polymer\utils\observer.py)`
- `[utils/state_features.py](c:\Users\HAMEDI\Desktop\RL_assisted_MPC_polymer\utils\state_features.py)`
- shared plotting core

### Reusable runners that need system/scenario metadata

- `[utils/mpc_baseline_runner.py](c:\Users\HAMEDI\Desktop\RL_assisted_MPC_polymer\utils\mpc_baseline_runner.py)`
- `[utils/horizon_runner.py](c:\Users\HAMEDI\Desktop\RL_assisted_MPC_polymer\utils\horizon_runner.py)`
- `[utils/matrix_runner.py](c:\Users\HAMEDI\Desktop\RL_assisted_MPC_polymer\utils\matrix_runner.py)`
- `[utils/weights_runner.py](c:\Users\HAMEDI\Desktop\RL_assisted_MPC_polymer\utils\weights_runner.py)`
- `[utils/residual_runner.py](c:\Users\HAMEDI\Desktop\RL_assisted_MPC_polymer\utils\residual_runner.py)`
- `[utils/combined_runner.py](c:\Users\HAMEDI\Desktop\RL_assisted_MPC_polymer\utils\combined_runner.py)`

### New distillation-specific modules likely needed

- `systems/distillation/config.py`
- `systems/distillation/adapter.py`
- `systems/distillation/scenarios.py`
- `systems/distillation/labels.py`
- `systems/distillation/data_io.py`

Residual does not need a distillation-native source notebook to be included. It just needs:

- the shared residual runner
- distillation scenario definitions
- distillation label metadata
- distillation data paths

## Plotting Transfer Rules

The plot generation should stay shared.

The following should become system metadata:

- output labels
- output units
- input labels
- input units
- time unit
- optional LaTeX display names

This will let the same plotting functions render:

- polymer plots in polymer notebooks
- distillation composition/temperature/reflux/reboiler plots in distillation notebooks

without copying plotting code again.

## Revised Transfer Phases

### Phase 1: Distillation foundation

- create `systems/distillation`
- add distillation adapter
- add data-path loader
- add scenario catalog
- add plot-label metadata
- wire plotting core to system metadata

### Phase 2: Baseline MPC

- create `distillation_MPCOffsetFree_unified.ipynb`
- normalize baseline result naming
- regenerate or remap nominal/ramp/fluctuation baseline outputs

### Phase 3: Single-agent RL notebooks

- horizon unified notebook
- matrix unified notebook with TD3 and SAC
- weight unified notebook with TD3 and SAC
- residual unified notebook ported from the shared main-repo residual modules

### Phase 4: Combined notebook

- create `distillation_RL_assisted_MPC_combined_unified.ipynb`
- use the current 4-agent combined architecture:
  - horizon
  - matrix
  - weight
  - residual

### Phase 5: Optional system-ID unification

- create `distillation_systemIdentification_unified.ipynb` later if you want the system-ID generation itself moved into the main repo

## Recommended Decisions Before Implementation

Please approve these explicitly before migration work starts:

1. Use `systems/distillation`, `Data/distillation`, and `Result/distillation`.
2. Keep one shared control flow, but let distillation use `ramp` and `fluctuation` disturbance profiles.
3. Include residual in the distillation migration scope.
4. Use the updated distillation system-ID notebook as a valid source notebook.
5. Normalize file naming during migration instead of preserving old mixed names.

## Recommended Immediate Next Step

If you approve this direction, the next implementation step should be:

1. create the distillation scaffolding under `systems/distillation`
2. wire plotting to system metadata
3. build `distillation_MPCOffsetFree_unified.ipynb`
4. normalize distillation baseline output naming
5. then port horizon, matrices, weights, residual, and combined

## Bottom Line

The distillation case now looks ready to migrate onto the main shared architecture.

The correct migration is:

- shared algorithms
- shared runners
- shared plotting
- distillation-specific plant/data/scenario/label modules
- new distillation unified notebooks

The updated system-ID and baseline MPC notebooks are now useful source notebooks.

The missing residual source notebooks are not a blocker anymore, because residual can be transferred from the already-unified main-repo implementation.
