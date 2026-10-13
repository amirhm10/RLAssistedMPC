# Polymer Scenario 2: ten parallel runs per category

The author requested ten concurrent runs after reviewing the computer's capacity. The wrapper now launches five SG and five plain/ungated repetitions together for each category. Categories remain sequential: horizon, weights, dynamic matrix (`markov`), residual, combined. The next category starts only after every current repetition has finished and saved. This includes the combined category.

The five paired seeds and all 50 complete 300-episode studies are retained. The change only combines the two mode groups in the parent scheduler and updates its progress output. Worker execution, mode selection, random seeding, training profile, output isolation, thread settings, failure handling, and resume semantics are preserved. A failure waits for the other nine repetitions before stopping progression; resume skips saved runs and retries failed jobs in new attempt directories.

Files inspected: `AGENTS.md`, `run_polymer_phase2_repeats.py`, `tests/test_polymer_phase2_repeats.py`, `POLYMER_PHASE2_REPEATS.md`, and the earlier wrapper change report. Files changed: the wrapper, its focused tests, its usage guide, and this new change report. No manuscript, controller, agent, shared defaults, plant, notebook, or existing simulation output was edited.

Verification: all eight focused tests passed. Real short-lived workers demonstrated ten-way overlap across SG and plain modes, separate output paths, the between-category save barrier, failure recovery, and preservation of completed runs. The dependency-free dry run shows five category groups of ten runs each. `git diff --check` passed. SHA-256 comparison confirmed all 117 other previously tracked files stayed unchanged; AST comparison also confirmed the worker, configuration adapter, job construction, dependency check, and saved-status functions stayed unchanged.

Full scientific training was not launched. These checks establish scheduling behavior, not peak RAM/GPU usage or throughput for ten actual training processes. Use the author's existing working scientific Python environment to start the study; no packages or environments were changed.
