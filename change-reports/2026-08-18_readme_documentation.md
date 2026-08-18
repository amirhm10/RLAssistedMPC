# README and Workflow Documentation Update

Date: 2026-08-18

## Scope

The repository landing documentation was rebuilt around the active manuscript and the tracked publication surface. No controller, agent, plant, or experiment behavior was changed.

## Changes

- Replaced the short README with a detailed description of the four RL assistance channels and critic-based supervisory gating.
- Documented the active polymer and distillation scenarios, default agent modes, entrypoints, dependencies, configuration switches, training schedule, outputs, validation commands, limitations, and citation status.
- Added a Mermaid architecture diagram and two workflow diagrams.
- Added a practical experiment guide with baseline first run order, preflight checks, monitoring, output interpretation, Aspen setup, and archival guidance.
- Copied the manuscript's exported overall framework PNG directly into `docs/figures/` without modification.
- Recorded figure provenance and its SHA-256 digest.
- Clarified that ignored historical code and generated result bundles are outside the public documentation evidence surface.

## Verification

- All documented relative paths and Markdown code fences were checked.
- The copied PNG SHA-256 matches the manuscript export.
- All 103 tracked Python files passed in-memory syntax compilation.
- The documented public snapshot test set passed with 87 tests.
