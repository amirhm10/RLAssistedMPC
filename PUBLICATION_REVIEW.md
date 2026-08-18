# Publication Scope Review

This guide explains how to verify that the publication branch preserves the latest method for both plants, both scenarios, and both agent modes without deleting local historical work.

## Preservation matrix

All cells in this matrix are retained in Git.

| Plant | Scenario 1 / Phase 1 | Scenario 2 / Phase 2 | Plain mode | SG mode |
|---|---|---|---|---|
| Polymer CSTR | Episodes 1–200 in `robustness_200_100` | Episodes 201–300 in `robustness_200_100` | DQN and TD3 | SG-DQN and SG-TD3 |
| Distillation column | Episodes 1–200 in `temperature_flip_200_100` | Episodes 201–300 in `temperature_flip_200_100` | DQN and TD3 | SG-DQN and SG-TD3 |

The scenarios and agent modes are independent axes. The scenario builders generate both phases before a selected plain or SG agent interacts with the shared runner.

## Where each decision is implemented

### Scenario definitions

- `systems/polymer/scenarios.py` contains `ROBUSTNESS_PHASE1_EPISODES`, `ROBUSTNESS_PHASE2_EPISODES`, the Phase-2 setpoints, both disturbance segments, and the combined 300-episode schedule.
- `systems/distillation/scenarios.py` contains `TEMPERATURE_FLIP_PHASE1_EPISODES`, `TEMPERATURE_FLIP_PHASE2_EPISODES`, both setpoint matrices, both feed-disturbance segments, and the combined 300-episode schedule.
- `utils/episode_profiles.py` validates the complete two-phase bundle used by every active runner.
- `utils/exploration_freeze.py` preserves the exploration-amplitude transition at the Scenario 1 to Scenario 2 boundary.

### Plain and SG modes

- `systems/polymer/notebook_params.py` retains plain and SG run profiles plus `resolve_polymer_combined_agent_kinds`.
- `systems/distillation/notebook_params.py` retains `resolve_distillation_agent_kind` and `resolve_distillation_combined_agent_kinds` for plain and SG selection.
- `DQN/` contains the plain DQN and SG-DQN implementations.
- `TD3Agent/` contains the plain TD3 and SG-TD3 implementations.
- `SACAgent/` remains tracked because the active continuous-runner import path supports SAC selections.

### Active controller families

Both plants retain the offset-free MPC baseline and the horizon, Markov/model, weight, residual, and combined RL entrypoints. Their complete transitive runtime import closure remains tracked under `Simulation/`, `systems/`, `utils/`, `BasicFunctions/`, `DQN/`, `TD3Agent/`, and `SACAgent/`.

## Automated evidence already checked

- The 12 active root entrypoints resolve to 87 retained runtime Python dependencies.
- All 16 focused tests remain tracked.
- No retained runtime or test file is matched by `.gitignore`.
- All 103 retained Python files compile in memory.
- The six retained identification and scaling files are valid pickle streams.
- No active Python implementation file was changed by the publication-scope commit.

The full pytest suite was not rerun in the publication shell because the documented `rl-env` interpreter was unavailable and the only available Python installation did not contain the scientific test stack.

## Review the branch in Git

From the repository root, run:

```powershell
git status --short --branch
git log --oneline --decorate -3
git diff --stat main...agent/publication-latest-runners
git diff --name-status main...agent/publication-latest-runners
git ls-files
```

Confirm that a required file is tracked and not ignored:

```powershell
git ls-files --error-unmatch systems/polymer/scenarios.py
git ls-files --error-unmatch systems/distillation/scenarios.py
git ls-files --error-unmatch DQN/supervisor_gated_dqn_agent.py
git ls-files --error-unmatch TD3Agent/supervisor_gated_agent.py
git check-ignore -v systems/polymer/scenarios.py
git check-ignore -v systems/distillation/scenarios.py
```

The `ls-files` commands must print each path. The `check-ignore` commands must print nothing.

Inspect the preserved scenario and mode symbols:

```powershell
git grep -n "ROBUSTNESS_PHASE1_EPISODES\|ROBUSTNESS_PHASE2_EPISODES"
git grep -n "TEMPERATURE_FLIP_PHASE1_EPISODES\|TEMPERATURE_FLIP_PHASE2_EPISODES"
git grep -n "combined_agent_mode\|resolve_distillation_agent_kind"
```

Confirm that ignored historical material still exists locally:

```powershell
Test-Path report
Test-Path archive
Test-Path Polymer\Data\mpc_results_dist_robustness_200_100.pickle
Test-Path Distillation\Data\mpc_results_disturb_fluctuation_temperature_flip_200_100.pickle
```

Each command should return `True`.

## Review in VS Code

Open Source Control and select the commit `Publish active Scenario 1 and 2 runners`. With GitLens, run `GitLens: Compare References`, choose `main` as the first reference, and `agent/publication-latest-runners` as the second. The comparison view lets you inspect every tracked removal while opening the unchanged local file beside it.

Use the Explorer to open the two scenario modules and the two notebook-parameter modules listed above. These four files provide the fastest human check that both plants, both scenarios, and both SG/plain choices are preserved.

## Decisions requiring human approval

Before remote publication, confirm these policy choices:

- Whether the six compact identification and scaling pickle files may be public
- Whether all historical reports, figures, result bundles, notebooks, and inactive algorithms should stay excluded
- Whether the repository needs a license and a reproducible environment specification before archival publication
- Whether proprietary Aspen Dynamics model files must remain external, as currently configured
