# Publication scope for the active Scenario 1/2 runners

## Outcome

The Git publication surface is restricted to the latest polymer `robustness_200_100` and distillation `temperature_flip_200_100` workflows. Both Scenario 1 / Phase 1 and Scenario 2 / Phase 2 are preserved for each plant. Their baseline, horizon, Markov, weight, residual, and combined entrypoints remain unchanged. Both supervisor-gated and plain-agent implementations remain available through the active configuration modules.

Historical reports, archives, presentations, prior result bundles, inactive algorithm families, unrelated utilities, editor settings, and Codex/AI-agent metadata are ignored and removed from Git tracking without deleting the local files.

## Retained implementation

- Twelve active root entrypoints for polymer and distillation
- The complete runtime import closure in `Simulation/`, `systems/`, `utils/`, `BasicFunctions/`, `DQN/`, `TD3Agent/`, and `SACAgent/`
- Focused automated tests for the active schedules and SG/plain execution paths
- Only the compact identification and scaling artifacts needed to initialize each plant model
- `PUBLICATION_REVIEW.md`, which maps every preserved scenario/mode axis to its implementation and provides independent Git and VS Code review commands

The Van de Vusse package under `systems/vandevusse/` remains tracked because the unchanged shared notebook setup imports its path helpers during module initialization. The separate Van de Vusse experiment tree is ignored.

## Generated data policy

MPC result pickles and RL result directories are ignored, including the latest local baseline bundles. They can be regenerated from the retained offset-free MPC entrypoints. No raw result or report file is deleted from the workstation.

## Validation target

Validation consists of import-closure auditing, Git ignore-boundary checks, in-memory compilation of every retained Python file, and the focused active test suite when the local scientific environment is available.
