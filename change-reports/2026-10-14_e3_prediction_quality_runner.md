# E3 prediction-quality runner, 2026-10-14

Implemented the author-approved E3 plan: four SG configurations (polymer/distillation, Markov/combined), one seed-7 training run per configuration, the original 300-episode Scenario 2 procedures, and one independent nominal OF-MPC probe per plant shared between its two configurations. This is runner development and technical verification, not a production launch or completed reviewer response.

The wrapper configures existing entrypoints in process-local defaults and compiles passive before/after hooks around their original plant-step statement. It preserves enclosing cleanup and omits later plotting/canonical baseline writes. Captures copy numeric predictor state, nominal matrices, scaling, LS acceptance, proposed/executed corrections, horizons, weights and residual context; they do not re-evaluate agents or branch the plant. Fixed predictors use identical actual future inputs and held-out future outputs at leads 1/4/8. Diagnostic Markov blocks extend to the requested lead without changing MPC horizons or its held-last input convention.

Default capture targets are 100 non-overlapping origins in each of four windows (episodes 14–20, 191–199, 201–210, 291–299), totaling 1,600 training origins. Nominal probes use two existing reference cycles, exclude the first cycle from scoring, and use 50 validation origins on the second. Both configurations share the same dataset hash. Frozen corrections come from the last preselected mature-original and adapted origin; causal observer histories are reconstructed without correction refitting or future measurement updates. These nominal results concern checkpoint predictors, not full-controller generalization.

The runner saves effective configuration, source/identification/Aspen hashes, environment versions, attempts, raw bundles, forecasts and residuals. Lead-specific per-output errors and normalized RMSE are kept separate from the reward diagnostic. Failed predictors remain in denominators. Independent analysis uses the saved scientific interpreter, produces new report directories, and does not retrain. Resume verifies primary artifacts and capture sidecars, skips valid stages, preserves failed attempts, and rejects a live previous worker. Existing Aspen file locks/COM compatibility are reused; worker-owned cleanup is idempotent and required before success. Native crashes/forced termination can leave cleanup unconfirmed and are recorded accordingly.

The implementation flags missing original identification-procedure details instead of reconstructing them from matrix dimensions. It also labels distillation temperature errors in native, unverified simulator units: the existing K label conflicts with negative configured values, so no conversion is inferred. Results remain descriptive for one training seed; gain corrections retain the nominal observer/free response and do not imply order/time-scale repair, causal combined-channel attribution, or guarantees.

## Files inspected

- Repository `AGENTS.md` and the research-result-loop, research-orchestrator, control-MPC, scientific-software-engineering and scientific-software-testing skills.
- Both plants' active Markov, combined and baseline Python entrypoints; their config, notebook defaults, scenarios, data I/O, labels and plant adapters.
- `utils/markov_runner.py`, `utils/combined_runner.py`, `utils/mpc_baseline_runner.py`, `utils/helpers.py`, `utils/observer.py`, `utils/observation_conditioning.py`, `utils/episode_profiles.py`, and notebook setup helpers.
- The existing polymer E1 runner, prediction/capture/runtime/smoke modules and tests; the distillation repetition wrapper and tests; environment specifications and `.gitignore`.
- Read-only paper context: the E3 review plan, R3.2 quotation, E1 implementation notes and run-recovery intake. No paper files were changed.

## Files added

- `run_e3_prediction_quality.py`.
- `review_experiments/e3/`: `__init__.py`, `common.py`, `capture.py`, `runtime.py`, `prediction.py`, `nominal.py`, and `analysis.py`.
- `tests/test_e3_prediction_quality.py`, `E3_PREDICTION_QUALITY_STUDY.md`, and this change report.

No existing scientific source, configuration, identification asset, result bundle, original figure, or manuscript was edited. The already dirty polymer repetition wrapper was preserved.

## Verification

Completed checks include the independent changing-input/nonzero-state forecast oracle; causal observers; nonfinite predictor accounting; fixed-window sampling; source/configuration preservation; standard-library-only dry run; interpreter/path preflight; parent failure/resume with real fake-worker subprocesses; live-worker rejection; fake Aspen cleanup/locking; nominal worker artifact hashing; real analysis subprocesses and non-destructive reanalysis; and plot rendering/inspection.

All 32 E3 cases pass across focused runs: 26 fast cases, four configuration-level training-preservation comparisons, and two shared-nominal validation cases. All 20 existing distillation repetition-wrapper checks also pass, for 52 targeted checks in total. The reduced distillation test-network defaults were aligned across the combined entrypoint's standalone parity checks; its production guard remains intact. Technical tests explicitly prohibit live Aspen COM application creation.

Full training parity compares trajectories, replay contents, learned network tensors, agent counters and Python/NumPy/PyTorch random states on the same compressed 300-episode clock. Small test networks run on CPU; production sizes and device selection are unchanged. No technical trajectory or plotted value is manuscript evidence. No production training or Aspen simulation was launched.

SHA-256 comparison confirms all 165 pre-existing tracked files retain their starting contents, including the already dirty repetition wrapper. All nine new Python files parse successfully. Read-only preflight verified the installed scientific interpreter and all three current Aspen model paths; representative polymer and distillation prediction/error plots were rendered and visually inspected. Ruff is not installed in the scientific environment; no tooling was installed for this task.
