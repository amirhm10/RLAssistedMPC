# RL-Assisted MPC: Active Publication Runners

This repository snapshot contains the latest active RL-assisted MPC workflow for two process-control plants. Both Scenario 1 and Scenario 2 are preserved for each plant. The same entrypoints support supervisor-gated (SG) and plain, non-SG agents.

In the implementation, Scenario 1 and Scenario 2 are named Phase 1 and Phase 2. Scenario selection is independent of the SG/plain agent-mode selection.

## Preserved scenarios

### Polymer CSTR

The polymer workflow controls viscosity and reactor temperature with coolant and monomer flows. Its active disturbed profile is `robustness_200_100`.

- Scenario 1, stored as Phase 1, covers episodes 1–200. It uses physical setpoints `[4.5, 324.0]` and `[3.4, 321.0]` with the legacy gradual disturbance schedule.
- Scenario 2, stored as Phase 2, covers episodes 201–300. It uses physical setpoints `[4.0, 321.5]` and `[3.3, 324.5]` with continued `Qi` and `Qs` changes and persistent fouling.

The active default is the plain-agent arm. SG-DQN and SG-TD3 remain selectable through `systems/polymer/notebook_params.py`.

### Distillation column

The distillation workflow controls tray-24 ethane composition and tray-85 temperature with reflux flow and reboiler duty. Its active disturbed profile is `temperature_flip_200_100`.

- Scenario 1, stored as Phase 1, covers episodes 1–200. It uses physical setpoints `[0.013, -23.0]` and `[0.028, -21.0]` with the established feed-fluctuation schedule.
- Scenario 2, stored as Phase 2, covers episodes 201–300. It uses physical setpoints `[0.013, -21.0]` and `[0.028, -23.0]` while continuing the same feed-disturbance random sequence.

The active default is the SG arm. Plain DQN and TD3 remain selectable through `systems/distillation/notebook_params.py`.

## Active entrypoints

| Controller family | Polymer | Distillation |
|---|---|---|
| Offset-free MPC baseline | `MPCOffsetFree_unified.py` | `distillation_MPCOffsetFree_unified.py` |
| Horizon supervisor | `RL_assisted_MPC_horizons_unified.py` | `distillation_RL_assisted_MPC_horizons_unified.py` |
| Markov/model supervisor | `RL_assisted_MPC_markov_unified.py` | `distillation_RL_assisted_MPC_markov_unified.py` |
| Weight supervisor | `RL_assisted_MPC_weights_unified.py` | `distillation_RL_assisted_MPC_weights_unified.py` |
| Residual supervisor | `RL_assisted_MPC_residual_unified.py` | `distillation_RL_assisted_MPC_residual_unified.py` |
| Combined supervisor | `RL_assisted_MPC_combined_unified.py` | `distillation_RL_assisted_MPC_combined_unified.py` |

The reusable implementations are under `Simulation/`, `systems/`, `utils/`, `DQN/`, `TD3Agent/`, and `SACAgent/`. The SG and plain implementations intentionally share the same active runners so that schedule, plant, reward, and comparison logic remain aligned.

## Data and generated results

The repository tracks only the compact system-identification and scaling artifacts required to initialize the two active workflows. Generated baselines, training bundles, plots, checkpoints, and historical reports are ignored.

Run the appropriate offset-free MPC entrypoint first to generate a matching baseline, then run an RL entrypoint. Polymer results are written under `Polymer/Results/` and distillation results under `Distillation/Results/`.

The distillation workflow requires a licensed Aspen Dynamics installation and the configured dynamic-model files. Those proprietary assets are not included.

## Environment and validation

The intended local Jupyter/Python environment is `rl-env`. The repository does not yet include a lockfile or environment specification, so dependency versions must be recorded separately before archival publication.

Focused checks for the active scenarios and SG/plain agent paths are under `tests/` and can be run with:

```powershell
python -m pytest tests
```
