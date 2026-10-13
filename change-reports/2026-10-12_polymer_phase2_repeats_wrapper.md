# Polymer Scenario 2 repetition wrapper — 2026-10-12

The author requested five independent repetitions of the existing polymer Scenario 2 study per category, with repetitions parallel and categories sequential, for both SG and plain modes. Existing working scientific code must remain unchanged.

Added `run_polymer_phase2_repeats.py`, `POLYMER_PHASE2_REPEATS.md`, and `tests/test_polymer_phase2_repeats.py`. The launcher executes the existing entrypoint sources in separate processes, applies only process-local experiment selection/seeds/output routing, records provenance, waits for every group member to save before progressing, and supports safe job-level resume.

Inspected the five `RL_assisted_MPC_*_unified.py` polymer entrypoints, `systems/polymer/notebook_params.py`, `systems/polymer/data_io.py`, `utils/notebook_setup.py`, plotting/export functions, DQN/TD3 RNG handling, the historical multiseed helper, and existing robustness-profile tests. The historical helper targets inactive variants and is not used.

The default plan is the unchanged complete `robustness_200_100` profile, five seeds, five categories, and both SG/plain modes: 50 runs in ten sequential groups, up to five concurrent processes. The large study is launched by the author through the wrapper; validation uses short synthetic workers rather than training. No package or environment change is part of this task.

Verification checks scheduling/save barriers, output isolation, failure/resume, worker-local configuration, and preservation of every previously tracked file against a pre-task SHA-256 snapshot. No scientific result is claimed from wrapper tests.

Completed verification: eight focused tests passed, including real-process five-way overlap, group ordering, worker creation failure, dependency preflight, and resume. All ten category/mode configurations were also checked against the actual NumPy-based shared defaults; all controller, reward, learning, and episode settings outside the declared wrapper selections remained equal. All 116 previously tracked files retained their original SHA-256 hashes.

The terminal's default Python has none of the required scientific packages, and the bundled artifact Python lacks SciPy, PyTorch, and Matplotlib. No packages were installed or environments changed. Full training was not launched during validation; run the wrapper using the author's existing working simulation interpreter, selected through the current environment or `--python`.
