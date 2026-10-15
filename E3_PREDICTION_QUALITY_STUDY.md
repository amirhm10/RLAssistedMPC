# E3: one-seed prediction quality on both plants

The runner evaluates polymer Markov, polymer combined, distillation Markov, and distillation combined. Each configuration trains once at seed 7 in SG mode. It preserves the existing 300-episode Scenario 2 procedure, including network sizes, warm-up, handoff, learning, exploration, gates, bounds, rewards, disturbances, and the final evaluation episode.

The primary question is whether the nominal linear predictor and its dynamic-matrix corrections predict the measured outputs under the tested operating conditions. Tracking reward is saved separately. These are descriptive single-seed diagnostics, without seed confidence intervals, guarantees, or causal attribution to individual combined channels.

## Commands

Run from the scientific repository with the existing `rl-assisted-env` interpreter. In PowerShell:

```powershell
$e3Python = 'C:\Users\hamediaa\AppData\Local\PythonEnvs\rl-assisted-env\python.exe'
& $e3Python run_e3_prediction_quality.py --dry-run
& $e3Python run_e3_prediction_quality.py --preflight
```

The first command uses only the standard library and creates nothing. Preflight imports the scientific dependencies and resolves/hashes the existing Aspen models without opening Aspen or creating a study. The current family mappings are baseline simulation2, Markov simulation11, and combined simulation9. `--aspen-root 'C:\path\to\Plant'` changes their root while preserving the resolved filenames and matching snapshot-directory convention.

Use `--aspen-number N` to select `C2S_SS_simulationN.dynf` and its matching snapshot directory for **all** distillation stages, including the nominal probe. The stages reuse that file sequentially; this overrides the family filenames only for the E3 study. The explicit root and number are saved with the study settings and cannot be changed on resume.

Full production command, **not launched during implementation**:

```powershell
& $e3Python run_e3_prediction_quality.py
```

Run polymer and distillation separately, keeping distillation on available simulation11 while simulations 1–10 are occupied:

```powershell
& $e3Python run_e3_prediction_quality.py --plants polymer
& $e3Python run_e3_prediction_quality.py --plants distillation --aspen-number 11
```

Each command runs both Markov and combined plus its plant's shared nominal probe. The two commands can run in separate terminals. Replace `11` with an available number from `12` through `16` if desired. Add `--categories markov` or `--categories combined` to restrict the controller family, or `--preflight` to check paths without starting training.

Every selected plant gets one shared nominal probe. Selecting a category does not multiply nominal probes. The default order is polymer nominal, polymer Markov, polymer combined, distillation nominal, distillation Markov, distillation combined. Workers run sequentially with one numerical thread. `--seed`, `--samples-per-window`, `--prediction-horizons`, `--threads`, `--python`, and `--output-dir` allow explicit alternatives; the complete effective configuration is saved.

The new study directory is printed at startup under `Results/e3/<UTC timestamp>_<id>`. To resume or regenerate derived analysis:

```powershell
& $e3Python run_e3_prediction_quality.py --resume 'C:\absolute\path\to\study'
& $e3Python run_e3_prediction_quality.py --analyse 'C:\absolute\path\to\study'
```

Resume uses the saved settings, interpreter, paths, model hashes, and code/data hashes. Completed stages are verified and skipped. Interrupted training restarts from its original initial condition in a new `a2`, `a3`, etc.; numeric predictor captures are not resumable training checkpoints. A live previous worker blocks duplication. Failed stages and logs remain intact. A failed analysis can be rerun without retraining or recollecting the nominal probe.

Aspen model locks use the same lock directory as the repetition wrapper. If another campaign owns a model, the E3 worker fails before opening it. Successful cleanup is required for accepting a distillation stage. Interrupts request Python cleanup at the next plant-step boundary. A native crash or forced termination can leave cleanup unconfirmed; inspect that worker's session before resuming. No code kills unrelated Aspen sessions.

## Training-trajectory diagnostics

There are 100 preselected, non-overlapping forecast origins per window and configuration:

| Window | Episodes | Interpretation |
| --- | --- | --- |
| release | 14–20 | Existing policy handoff/release protocol |
| prechange_mature | 191–199 | Mature original operating regime, including its original disturbances |
| plant_change | 201–210 | Immediate changed-condition behavior |
| adapted | 291–299 | Late adaptation before the final evaluation episode |

Origins are chosen using a local RNG independent of Python/NumPy/PyTorch training streams, before inspecting outcomes. Each complete future remains inside its specified window. The default study therefore saves 1,600 training origins across four configurations.

At each origin, all four predictors share the same decision-time nominal observer state, free response, scaling, and subsequent **actually applied physical inputs**, including residual input changes in combined mode:

- nominal: zero dynamic-matrix correction;
- accepted_ls_or_zero: the accepted bounded LS correction, or zero if rejected;
- rl_proposed: the recorded bounded RL proposal, even when rejected by supervision;
- executed: the correction actually used after gates, projections, and fallback.

Candidate coefficients and initial states are copied before advancing the plant. Only numeric records are captured; the code does not ask agents for additional proposals or perform counterfactual plant rollouts. When the eight future inputs and outputs become available, the forecasts use the common input sequence and compare with the outputs. Future outputs do not refit the correction, initialize the forecast, or update its observer.

The diagnostic is a retrospective **known-input forecast**, not MPC's planned-input, held-last forecast. The full Toeplitz map represents every realized future input. Markov blocks and their correction basis are generated from the same matrices out to eight steps, including when distillation's controller horizon is six or its combined horizon is four. This extends only the diagnostic calculation.

## Independent nominal validation

For each plant, initialize a fresh nominal plant, disable imposed disturbances, and execute the existing nominal OF-MPC for two cycles of its existing reference pair. The first cycle supplies causal observer history and is excluded from scoring. The second provides 50 non-overlapping eight-step origins, split equally between the two reference holds. Default lengths are 1,600 total control steps for polymer and 800 for distillation.

Both controller configurations use the exact same saved physical inputs and outputs, identified by the nominal dataset's SHA-256. Each uses the correction sets from its **last preselected origin** in `prechange_mature` and `adapted`. These checkpoints are chosen without ranking prediction errors. No correction is refit on nominal validation data, and no trained controller is run on that dataset.

For each checkpoint, reconstruct causal nominal observer states from the probe measurements using that checkpoint's original observer convention and gain. Every predictor at a validation origin receives the same state. Thus nominal results assess the frozen checkpoint predictors, rather than generalization of the full learned control policy. The dataset is reused rather than counting repeated calculations as independent experiments.

## Artifacts and interpretation

Each stage has `job.json`, `status.json`, `worker.log`, `runtime_metadata.json`, and either `training.pkl` plus `capture/`, or `nominal.npz` plus nominal metadata. Capture files contain initial states, matrices, scaling, correction vectors, actual future inputs/outputs, all four forecasts, source codes, and failed-predictor flags. File hashes protect both primary artifacts and diagnostic sidecars on resume. Original identification assets and Aspen model hashes are recorded.

Derived reports are saved to a fresh `analysis/<UTC timestamp>_<id>/` directory:

- `prediction/prediction_errors.csv`: lead-specific physical RMSE, MAE, signed bias, maximum absolute error, normalized RMSE, and valid/failed origin counts, separately for each output, configuration, dataset, window, and predictor;
- `prediction/residuals.csv`: signed errors at each origin and lead;
- `prediction/source_fractions.csv`: recorded execution-source fractions, with nominal entries explicitly referring to the source at the frozen training checkpoint;
- prediction/residual examples from the first preselected origin and aggregate error-versus-lead plots;
- nominal-checkpoint records sharing the original nominal dataset hash;
- `tracking_reward_separate.csv`: a separate final-episode reward diagnostic with its source bundle.

One-step means lead 1; four-step means lead 4; eight-step means lead 8. The metrics average across origins at the specified lead, **not across the prefix from 1 through that lead**. Normalization uses the fixed identification output range, not a statistic estimated from validation outcomes. Failed/nonfinite predictors remain in the denominator and receive explicit failed counts.

Polymer viscosity is recorded in L/g and reactor temperature in K. The existing distillation metadata labels temperature as K while its configured values are negative; this runner does not infer or convert the native Aspen unit. Distillation temperature errors are labeled as native simulator units, unverified, until the simulator convention is checked.

Model orders, sample period, scaling, correction basis/bounds, observer convention, source hashes, and effective defaults are saved. Original excitation, order-selection rationale, and fitting/validation details cannot all be reconstructed from the stored matrices and are explicitly flagged for verification from the identification records. Running this code alone does not complete the self-contained identification description requested by R3.2.

Dynamic-matrix gain corrections retain the nominal observer/free response. They do not generally repair wrong model order, changed delays/time scales, or arbitrary nonlinear operating dependence. Better combined tracking does not establish better prediction accuracy or identify a causal contribution from another channel; E4 remains separate.

## Technical checks

```powershell
& $e3Python -m pytest tests/test_e3_prediction_quality.py -q --basetemp=.tmp_e3_verification
& $e3Python run_e3_prediction_quality.py --smoke --categories markov
```

The first command includes synthetic forecast oracles, observer causality, unchanged-training comparisons, saved-data analysis, failure/resume, and fake plant/COM resource contracts. Distillation integration uses a deterministic fake physical plant and never opens Aspen. Parity fixtures use reduced networks on both sides only to limit test cost.

CLI smoke is polymer-only, retains the original network sizes, compresses the 300-episode clock, samples one origin per window, and uses short nominal cycles with two nominal validation origins. Every artifact is labeled as technical smoke. Neither these checks nor any resulting plots constitute manuscript performance evidence.
