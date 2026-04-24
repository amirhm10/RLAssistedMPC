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
- For any new notebook, notebook-default family, runner config, or similar execution surface you add, default polymer runs to `run_mode = "disturb"` and default distillation runs to `run_mode = "disturb"` with `disturbance_profile = "fluctuation"`, unless the user explicitly asks for another default. Keep nominal support in the code and config tables unless the user asks to remove it.
- For smoke validation of shared unified runner changes, use polymer notebook executions as the default runtime confirmation path. Do not open the distillation Aspen column just for smoke tests unless the user explicitly asks for a full distillation execution.
- In `systems/distillation/config.py`, keep Aspen family mapping generic by notebook family, not split by RL algorithm. Use one simulation number per family and map all disturbance profiles for that family to the same `C2S_SS_simulationN.dynf`. The current sequence is `system_id=1`, `baseline=2`, `horizon=3`, `horizon_dueling=4`, `matrix=5`, `structured_matrix=6`, `weights=7`, `residual=8`, `combined=9`, `reidentification=10`. If a new distillation method family is added later, assign the next unused number instead of branching by TD3/SAC.
- Treat dirty notebooks, tracked result images, and timestamped output folders as user-owned work. Do not clean, rename, or regenerate them unless the user asks.
- Avoid editing timestamped artifact directories under `Data/`, `Result/`, and `Results/` by default.
- Do not treat generated plots, `.pkl` bundles, or cached `.pyc` files as the primary implementation surface.
- Notebook execution order matters. Several notebooks define helper functions in cells and rely on globals such as `A_aug`, `B_aug`, `IC_opt`, `bnds`, `cons`, reward functions, and decision-interval constants.
- When tracing an experiment, read the notebook cells that define the supervisor function before assuming the reusable module layer is complete.
- When comparing outputs, use the run-folder naming convention plus `input_data.pkl` to identify provenance instead of guessing from image names alone.
- If a task asks for "the training loop" or "current experiment logic," verify whether the notebook-local supervisor is the active implementation before editing module-level helpers.
- For major repo changes, add a short markdown note under `change-reports/` and include it in a local git commit. For normal completed changes, make a local git commit without asking after verification. Stage only files relevant to the current task; never include unrelated dirty notebooks, generated artifacts, cache files, or timestamped experiment output unless the user explicitly asks. Do not push to `origin`/GitHub remote until the user asks for the end-of-day remote push.

## Environment Caveats

- There is no repo-level `README`, `pyproject.toml`, `requirements.txt`, or environment file in the current tree.
- Do not claim the repo has a reproducible shell environment unless the user adds one.
- The intended notebook/runtime environment is the Conda environment `rl-env`.
- The Jupyter kernel name for this repo should be `rl-env` with display name `Python (rl-env)`.
- When running Python-based checks from the terminal, prefer `C:\Users\HAMEDI\miniconda3\envs\rl-env\python.exe` instead of the shell `python`, because the shell interpreter may not have the scientific stack installed.
- Because the environment is not declared in-repo, prefer documenting observed behavior over inventing setup commands.

## VS Code Workflow Preferences

- Assume the user is working in VS Code with the Python, Pylance, Jupyter, Ruff, Markdown Preview Enhanced, Markdown All in One, markdownlint, GitLens, GitHub Pull Requests and Issues, Rainbow CSV, Data Wrangler, LaTeX Workshop, Catppuccin, and Material Icon Theme extensions available.
- When giving IDE workflow guidance, prefer actions that use those extensions: Jupyter kernel selection and notebook outline for `.ipynb`, Markdown Preview Enhanced for reports, Ruff for Python linting/formatting, GitLens for change history, Data Wrangler or Rainbow CSV for tabular data, and LaTeX Workshop for `.tex` report builds.
- For Markdown reports with equations, write display math in renderer-friendly LaTeX. Prefer one-line `$$ ... $$` blocks for single equations, for example `$$ \Delta R_{\mathrm{release}} = \bar{R}_{\mathrm{live},1{:}10} - \bar{R}_{\mathrm{warm},\mathrm{tail}}. $$`. Use `\begin{aligned}...\end{aligned}` only when there are multiple aligned equations, keep each relation's operator and right-hand side on the same visual line, use `\\` for line breaks, and use `\mathrm{...}` for text-like subscripts or superscripts such as `cand`, `nom`, `phys`, `raw`, and `eff`. Avoid loose multi-line `$$` blocks with separate operator-only lines such as `=`, `&=`, `-`, or equations joined only by `\qquad`.
- Codex chat drag-and-drop and raw Windows paths may be unreliable. Prefer repo-relative paths in backticks, such as `report/distillation_warm_start_training_analysis.md` or `distillation_RL_assisted_MPC_structured_matrices_unified.ipynb`, when referring to files in chat.
- During longer interactive work, occasionally include one short practical explanation of a relevant VS Code, notebook, Markdown, GitLens, or data-inspection feature so the user can learn the workflow while the repo work progresses.

## Known Codebase Caveats

- Generated artifacts dominate the tree. There are far more plots and pickles than source files.
- Logic is duplicated across reusable modules and notebooks. Do not assume one copy has been fully migrated into the other.
- `Simulation/rl_sim.py` appears legacy or stale relative to the current TD3/SAC/DQN agent APIs. It still references patterns such as `agent.replay_buffer.add`, `agent.train(...)`, and `agent.exploration_noise_std`, while the current agent modules use `buffer`, `push`, `train_step`, and newer scheduling logic.
- `BasicFunctions/plot_fns.py` also reflects an older plotting/output style than `utils/plotting.py`.
- Git history is minimal, so experiment provenance is easier to reconstruct from notebook families and timestamped run folders than from commits.
- The current worktree is dirty. Assume some notebooks and result artifacts are mid-experiment, not finalized baselines.
- The archived distillation subtree at `DIstillation Column Case/RL_assisted_MPC_DL/` should be treated as a read-only reference during migration. Do not extend it when adding new distillation work; add new code under `systems/distillation/`, `Data/distillation/`, `Result/distillation/`, and the root `distillation_*_unified.ipynb` entrypoints instead.
