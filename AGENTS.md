# AGENTS.md

## Project Intent

This repository studies RL-assisted MPC across two canonical workflows:

- polymer CSTR
- distillation column

- The polymer nonlinear plant is the reactor model in `Simulation/system_functions.py`.
- The distillation nonlinear plant adapter is in `systems/distillation/plant.py`.
- Polymer controlled outputs are viscosity (`eta`) and reactor temperature (`T`), with manipulated inputs coolant flow `Qc` and monomer flow `Qm`.
- Distillation controlled outputs are tray-24 ethane composition and tray-85 temperature, with manipulated inputs reflux flow and reboiler duty.
- The project uses system identification data and scaling artifacts in `Data/` to build a linear offset-free MPC model, then layers RL supervisors on top of that baseline.
- Current RL families in the repo are TD3, SAC, and DQN.

In practice, the research flow is:

1. Generate or inspect step-test data and identification artifacts.
2. Run baseline offset-free MPC.
3. Train or evaluate RL agents that adjust horizons, model multipliers, weights, or combined supervisory actions.
4. Save timestamped result bundles under `Data/`, `Result/`, and `Results/`.

## Source Map

Treat the repo as three layers: reusable modules, notebook experiment entrypoints, and generated artifacts.

### Reusable Modules

- `Simulation/system_functions.py`
  Nonlinear polymer CSTR plant model.
- `Simulation/sys_ids.py`
  Step-test generation, simulation helpers, CSV preparation, and system identification utilities.
- `Simulation/mpc.py`
  Offset-free MPC solver, observer-gain helpers, baseline MPC rollout functions, and disturbance-aware MPC variants.
- `TD3Agent/`
  Continuous-control TD3 actor, critic, replay buffer, and checkpoint helpers.
- `SACAgent/`
  Continuous-control SAC implementation built on the shared critic structure.
- `DQN/`
  Discrete-action DQN used for horizon-selection experiments.
- `utils/helpers.py`
  Shared scaling helpers, augmented system loading, horizon recipe helpers, and disturbance schedule helpers.
- `utils/helpers_net.py`
  Shared neural-network builder and activation initialization utilities.
- `utils/plotting.py`
  Current plotting and comparison entrypoints for TD3, SAC, DQN, and multi-agent runs.
- `systems/distillation/`
  Distillation-specific canonical layer for plant access, data I/O, scenario generation, label metadata, and system-identification helpers.
- `BasicFunctions/`
  Older helper code and plotting/pretraining utilities. Useful for legacy context, but not the first place to extend current experiments.

### Notebook Entrypoints

The root `.ipynb` files are not just reports. Many contain experiment-local supervisor functions that are still the real execution entrypoints for current research runs.

- `systemIdentification.ipynb`
  System-identification workflow and preprocessing.
- `MPCOffsetFree*.ipynb`
  Baseline offset-free MPC experiments, including disturbance variants.
- `RL_assisted_MPC_horizons*.ipynb`
  DQN horizon-selection experiments.
- `RL_assisted_MPC_matrices*.ipynb`
  TD3 and SAC multiplier experiments, including disturbance and mismatch variants.
- `RL_assisted_MPC_weights*.ipynb`
  TD3 weight-tuning experiments.
- `RL_assisted_MPC_Poles.ipynb`
  Observer-pole study.
- `RL_assisted_MPC_combined*.ipynb`
  Multi-agent supervisor experiments that combine horizon, model, and weight decisions.
- `RL_assisted_MPC_*model_mismatch*.ipynb`
  Model-mismatch and residual-policy variants.
- `distillation_*_unified.ipynb`
  Canonical distillation notebook entrypoints built on the shared runner and plotting layer.

Notebook-local functions currently matter. Examples discovered in the repo include:

- `run_dqn_mpc_horizon_supervisor`
- `run_rl_train_disturbance_gradually`
- `run_multi_agent_rl_mpc`
- residual mismatch helpers such as `compute_band_scaled` and `residual_outside_weight`

### Generated Artifacts

Assume these directories are outputs unless the user explicitly says otherwise.

- `Data/`
  Baseline MPC results, stored pickles, scaling artifacts, and many single-agent experiment runs.
- `Data/distillation/`
  Canonical distillation system-identification assets, baseline MPC pickles, and distillation experiment runs.
- `Result/`
  Comparison plots across experiments.
- `Result/distillation/`
  Canonical distillation comparisons and plot outputs.
- `Results/`
  Combined multi-agent outputs and comparisons.

Common naming patterns:

- `Data/mpc_result_*/<timestamp>/`
  Baseline MPC plots for a scenario.
- `Data/horizon_*/<timestamp>/`
  DQN horizon-selection runs.
- `Data/td3_weights_*/<timestamp>/`
  TD3 weight-tuning runs.
- `Data/td3_multipliers_*/<timestamp>/`
  TD3 model-multiplier runs.
- `Data/sac_multipliers_*/<timestamp>/`
  SAC multiplier runs.
- `Result/*compare*/<timestamp>/`
  Cross-run comparisons.
- `Results/multi_agent_run_*/<timestamp>/`
  Combined supervisor outputs.

Most run folders contain `input_data.pkl` plus derived `.png` figures.

## Experiment Map

Use this map when locating the active notebook entrypoints.

- `systemIdentification.ipynb`
  Generates and analyzes step-test data used for model identification and scaling.
- `MPCOffsetFree_unified.ipynb`
  Canonical offset-free MPC baseline with shared `RUN_MODE = "nominal" | "disturb"`.
- `distillation_systemIdentification_unified.ipynb`
  Canonical distillation system-identification workflow.
- `distillation_MPCOffsetFree_unified.ipynb`
  Canonical distillation offset-free MPC baseline with `RUN_MODE` and `DISTURBANCE_PROFILE`.
- `RL_assisted_MPC_horizons_unified.ipynb`
  Canonical DQN horizon-selection workflow with shared nominal/disturbance handling.
- `distillation_RL_assisted_MPC_horizons_unified.ipynb`
  Canonical distillation DQN horizon-selection workflow with distillation-specific disturbance profiles.
- `RL_assisted_MPC_matrices_unified.ipynb`
  Canonical matrix-multiplier workflow with TD3/SAC selection and shared mismatch-state mode.
- `distillation_RL_assisted_MPC_matrices_unified.ipynb`
  Canonical distillation matrix-multiplier workflow with TD3/SAC selection and shared mismatch-state mode.
- `RL_assisted_MPC_weights_unified.ipynb`
  Canonical penalty-multiplier workflow with TD3/SAC selection and shared mismatch-state mode.
- `distillation_RL_assisted_MPC_weights_unified.ipynb`
  Canonical distillation penalty-multiplier workflow with TD3/SAC selection and shared mismatch-state mode.
- `RL_assisted_MPC_residual_unified.ipynb`
  Canonical residual-correction workflow with TD3/SAC selection, mismatch-state mode, and optional `rho` authority.
- `distillation_RL_assisted_MPC_residual_unified.ipynb`
  Canonical distillation residual-correction workflow with TD3/SAC selection, mismatch-state mode, and optional `rho` authority.
- `RL_assisted_MPC_combined_unified.ipynb`
  Canonical four-agent supervisor combining horizon, matrix, weight, and residual agents.
- `distillation_RL_assisted_MPC_combined_unified.ipynb`
  Canonical distillation four-agent supervisor combining horizon, matrix, weight, and residual agents.
- `RL_assisted_MPC_Poles.ipynb`
  Observer-pole variation study.

Legacy split notebooks were removed after the unified migration. Historical mismatch behavior is documented in `report/model_mismatch_usage.md` rather than preserved as active notebook entrypoints.

## Working Rules For Agents

- Prefer edits in reusable `.py` modules when the behavior is shared across experiments.
- Only edit notebooks when the logic truly lives in notebook cells and is not already represented in a reusable module.
- Treat dirty notebooks, tracked result images, and timestamped output folders as user-owned work. Do not clean, rename, or regenerate them unless the user asks.
- Avoid editing timestamped artifact directories under `Data/`, `Result/`, and `Results/` by default.
- Do not treat generated plots, `.pkl` bundles, or cached `.pyc` files as the primary implementation surface.
- Notebook execution order matters. Several notebooks define helper functions in cells and rely on globals such as `A_aug`, `B_aug`, `IC_opt`, `bnds`, `cons`, reward functions, and decision-interval constants.
- When tracing an experiment, read the notebook cells that define the supervisor function before assuming the reusable module layer is complete.
- When comparing outputs, use the run-folder naming convention plus `input_data.pkl` to identify provenance instead of guessing from image names alone.
- If a task asks for "the training loop" or "current experiment logic," verify whether the notebook-local supervisor is the active implementation before editing module-level helpers.
- For major repo changes, add a short markdown note under `change-reports/`, include it in the commit, and push the result to `origin`.

## Environment Caveats

- There is no repo-level `README`, `pyproject.toml`, `requirements.txt`, or environment file in the current tree.
- Do not claim the repo has a reproducible shell environment unless the user adds one.
- The intended notebook/runtime environment is the Conda environment `rl-env`.
- The Jupyter kernel name for this repo should be `rl-env` with display name `Python (rl-env)`.
- When running Python-based checks from the terminal, prefer `C:\Users\HAMEDI\miniconda3\envs\rl-env\python.exe` instead of the shell `python`, because the shell interpreter may not have the scientific stack installed.
- Because the environment is not declared in-repo, prefer documenting observed behavior over inventing setup commands.

## Known Codebase Caveats

- Generated artifacts dominate the tree. There are far more plots and pickles than source files.
- Logic is duplicated across reusable modules and notebooks. Do not assume one copy has been fully migrated into the other.
- `Simulation/rl_sim.py` appears legacy or stale relative to the current TD3/SAC/DQN agent APIs. It still references patterns such as `agent.replay_buffer.add`, `agent.train(...)`, and `agent.exploration_noise_std`, while the current agent modules use `buffer`, `push`, `train_step`, and newer scheduling logic.
- `BasicFunctions/plot_fns.py` also reflects an older plotting/output style than `utils/plotting.py`.
- Git history is minimal, so experiment provenance is easier to reconstruct from notebook families and timestamped run folders than from commits.
- The current worktree is dirty. Assume some notebooks and result artifacts are mid-experiment, not finalized baselines.
- The archived distillation subtree at `DIstillation Column Case/RL_assisted_MPC_DL/` should be treated as a read-only reference during migration. Do not extend it when adding new distillation work; add new code under `systems/distillation/`, `Data/distillation/`, `Result/distillation/`, and the root `distillation_*_unified.ipynb` entrypoints instead.
