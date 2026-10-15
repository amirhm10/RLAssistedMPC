# Polymer E1 matched-rollout study

## Remaining-run recovery, 2026-10-15

Four standalone channels have 1,600 validated matched decisions in
`Polymer/Results/e1/20261013_192015_916c06`. Residual reached cycle 300 and
saved all 400 pairs, but an overly broad final source check blocked its main
trajectory export. Its frozen inference snapshots do not contain optimizer or
replay state for exact training resumption. Retain these pairs without rerunning
residual. The combined capture is still missing.

`run_polymer_review_recovery.py` audits the retained records and schedules only
combined E1 (300 cycles; the original 1,600 remaining pairs), then missing E2 SG
(200 cycles). It preserves all old folders/statuses and the original settings.
Default invocation is a read-only plan; `--run` starts production. New results
go under `Polymer/Results/review_recovery/`. Live worker logs appear in CMD.

```cmd
python run_polymer_review_recovery.py --dry-run
run_remaining_polymer_reviews.cmd
```

The source fingerprint now excludes unrelated `review_experiments/e3/` files
while retaining checks on shared implementation and E1 dependencies. Raw main
trajectories are saved before final integrity checks; a failed status still
excludes them from automatic verified reuse. Historical manifests are not
rewritten to conceal failed exports. Do not use `--resume` on the old E1
manifest after this documented wrapper revision; use the recovery launcher.

Author-approved implementation scope, 2026-10-13: use **one training seed by default**, reuse the existing training/controller code, preserve the paper's warm-up and hand-off, and add the capture/evaluation needed for reviewer evidence. Full E1 production training has not been launched during development.

## Default study

`run_polymer_e1.py` uses seed **7** and the existing `robustness_200_100` Scenario 2 profile. It runs horizon, weights, dynamic matrix (`markov`), residual, then combined. Each configuration still trains through its original 300 episodes. The combined configuration selects SG-DQN and three SG-TD3 agents through the existing mode resolver.

| Setting | Default |
|---|---|
| Training seeds | 1: seed 7 |
| Capture/training runs | 5: one for each configuration |
| Channels examined | All four individually, and all four in combined |
| Decisions per channel/window | 100, sampled without filtering by gate acceptance |
| Windows | Release 14–20; mature pre-change 191–199; change response 201–210; late adaptation 291–299 |
| Matched channel decisions | 3,200 |
| Primary rollout | 200 control steps; return/physical metrics at 20, 80, 200 |
| Longer-rollout subset | 10% per channel/window, extended to 800; also report 400 |
| Reusable full snapshots | 1 per channel/window: 32 across the complete study |

The pre-change window retains the original training disturbance schedule; it should not be described as a disturbance-free nominal experiment. Sampling is constrained to actual gate decision times. Horizon choices retain their four-control-step hold. The other current gates act each control step.

The primary branch budget is 1.28 million simulated control steps; extending the 10% subset adds 384,000. The five main training trajectories add 1.2 million. These step counts are not wall-clock estimates.

One seed supports conditional decision diagnostics for these learning trajectories. Hundreds of points from one trajectory do not constitute independent training replications. The separately requested repeated Scenario 2 campaign supplies a different performance-replication dataset. This E1 study does not replace that campaign or unseen-reference evaluation.

## Preservation contract

The full-study adapter changes only run/mode selection, seed and output routing. Network architecture, optimizer, reward, adaptation bounds, gate margins, learning schedules, warm-up duration, critic-only period, policy hand-off, operating-change profile, final evaluation episode and solver calculations come from the existing entrypoints/defaults.

The active SG defaults use the existing 10-episode warm start and additional three-episode hand-off/critic-only window. Their exact step boundaries are recorded from the running code, rather than recreated in a new training algorithm.

No existing controller, agent, plant, shared-default, or phase-2 repetition-wrapper source file is edited. New Python utilities live under `review_experiments/e1/`, outside the source directories hashed by the running repetition wrapper.

The adapter privately compiles the existing runner AST, exposes its local state in a private namespace, and adds suspension points around existing gate-result assignments. Every original training statement runs in its original order. This enables an exact mid-step snapshot even in combined, where the dynamic-matrix critic is updated before its gate call. Counterfactual continuations execute the same original suffix and subsequent control-step statements. Unsupported gate nesting or snapshot restoration inconsistencies fail explicitly.

Tests compare original and instrumented trajectories, replay data and available final network parameters. The shortened technical clock exercises warm-up, hand-off, the episode-201 change, ongoing learning and episode-300 evaluation. It is not a full native-clock production validation.

The software parity fixtures use two 32-unit layers and an eight-sample minibatch solely to keep the checks short and exercise critic updates within the compressed hand-off window. Neither adjustment exists in the production defaults adapter. Separate CLI checks use the original five-layer, 512-unit networks.

## Matched branches

At each sampled gate, save the actual exploratory policy proposal and exact supervisor proposal, critic values, score, gate configuration, observation and decision-time replay-support diagnostics.

The two branches start with the same nonlinear plant, estimator, controller memory, least-squares history, previous actions, networks and event/reference clock. Force the policy candidate in one branch and the supervisor candidate in the other, then use the existing action mapping, bounds, guards and solver behavior. A forced candidate can therefore still be altered by a later existing guard; executed adaptation/input logs are retained.

Future branch proposals are deterministic and networks/replay are frozen. Running observation-normalizer statistics are frozen in branches; the observer, plant and data-dependent least-squares supervisor continue evolving. Both branches use the same continuation rule, with their own resulting states. They do not receive identical future input commands.

Main training is suspended during the pair, then receives its exact original Python/NumPy/Torch/CUDA random state and network modes back. Branch transitions never enter training replay. Main training itself remains live and uses its original exploration and updates.

The primary continuation is `gated`. Optional `--continuation policy` or `--continuation supervisor` forces future candidate source choices while preserving the remaining controller calculations. `policy` uses the online actors; it is not an implementation of TD3 target-actor smoothing or an exact Bellman-target evaluation.

Critic targets use actor-based bootstrapping, while the actual gated continuation differs. Horizon replay also uses one-step transitions despite a four-step action hold. Accordingly, report finite-horizon action-ranking/selection usefulness, not an unbiased infinite-horizon Q estimate or calibrated confidence guarantee.

## Added reviewer diagnostics

- **E1 / Reviewer 2.1:** harmful acceptance, beneficial rejection, return regret, raw-critic ranking agreement, phase breakdown, decision-time replay support and twin-critic disagreement. Failed branches and solver outcomes are retained.
- **E3 / Reviewer 3.2:** at standalone and combined dynamic-matrix decisions, compare nominal, accepted-LS/zero and RL-proposed predictors at 1, 4 and 8 steps. Every predictor uses the same initial observer state and observed future input sequence within a branch. Coefficients are fixed at the decision; future outputs only calculate errors. All predictors retain the nominal free response. These retrospective open-loop forecasts do not use MPC's held-last planned-input assumption and do not represent an independent identification experiment or the distillation case.
- **E4 / Reviewer 2.2 and Reviewer 3.6:** individual-versus-combined conditional channel effects and snapshots containing all combined-controller context. A full joint/16-combination experiment is a later extension; these results do not establish independent causal channel contributions.
- **E5 / Reviewer 3.9:** reanalyse the same action pairs with margin offsets to examine local decision sensitivity. The continuation is held at its original settings, so this is not a full closed-loop margin sweep. Disturbance/order/calibration stress tests remain separate work.

Physical metrics use viscosity in L/g, temperature in K and input flows in L/h, as specified by the current polymer metadata. Return uses the original reward and original gamma per control step. A positive `--return-tolerance` should be agreed before reporting practically meaningful decision-error rates; zero gives unthresholded descriptive diagnostics.

## CMD commands

Activate the existing environment and inspect the plan:

```bat
call "C:\ProgramData\anaconda3\Scripts\activate.bat" rl-assisted-env
python "C:\Users\hamediaa\OneDrive - McMaster University\PythonProjects\RL_assisted_MPC\run_polymer_e1.py" --dry-run
```

Start the full default study when ready:

```bat
python "C:\Users\hamediaa\OneDrive - McMaster University\PythonProjects\RL_assisted_MPC\run_polymer_e1.py"
```

`--seeds 7 11` adds a seed without changing scientific sources. Worker/thread controls are explicit; the default is one worker and one Torch/BLAS thread. Multiple seeds within a category can run in parallel; the next category waits for all current-category jobs to save. With one seed, category training remains sequential even if the worker limit is higher. The same interpreter runs every child. CUDA is used when available, as in the existing entrypoints.

Technical checks only:

```bat
python "C:\Users\hamediaa\OneDrive - McMaster University\PythonProjects\RL_assisted_MPC\run_polymer_e1.py" --smoke
```

Smoke first builds the original 300 x 800-step profile, then applies a process-local compressed clock for a cheap integration check. Episode order, warm-up/hand-off episode counts, change at 201 and evaluation at 300 are retained. Every record is labelled `smoke`; its performance numbers must not enter the paper. Full mode applies no clock compression or network resizing.

Reanalyse a completed batch and calculate local margin sensitivity:

```bat
python "C:\Users\hamediaa\OneDrive - McMaster University\PythonProjects\RL_assisted_MPC\run_polymer_e1.py" --analyse "PATH_TO_E1_BATCH" --margin-offsets 0 0.25 0.5 1
```

Offsets are in the existing reward/value-score units. Summary rows remain separate by training seed. No seed-level confidence intervals are invented from repeated decisions.

Replay a selected snapshot:

```bat
python "C:\Users\hamediaa\OneDrive - McMaster University\PythonProjects\RL_assisted_MPC\run_polymer_e1.py" --replay-snapshot "PATH_TO_SNAPSHOT.pkl.gz"
```

Snapshots contain CPU inference weights, full plant/controller state and the original candidate pair. Replay uses CUDA when available. They omit optimizers and mutable replay contents: they resume frozen validation, not online training. Scientific source/data hashes must match. Load only locally generated trusted snapshots.

## Outputs and restart behavior

New bundles are under `Polymer/Results/e1/<timestamp_uuid>/`. Each category/seed/attempt has:

- `job.json`, `status.json`, `worker.log` and source hashes;
- `training.pkl`, the standard training result bundle;
- `capture/sampling.json`, timing and preselected decision indices;
- `capture/decisions.jsonl`, proposals/scores/support/returns and prediction diagnostics;
- compressed paired traces and selected reusable snapshots;
- `analysis/summary.csv`, `prediction_summary.csv`, and `summary.json` with failures/limitations.

`--resume PATH_TO_BATCH` skips completed jobs and restarts an unfinished job as a new complete training attempt. It does not restore online learning mid-trajectory. Source changes block batch resume. Previous attempts are retained; batch analysis includes only the latest completed attempt for each job and refuses duplicate decision identities.

Development artifacts are distinct from production evidence. Original result folders, the original manuscript and its frozen files are preserved.

## Development verification

Original-versus-instrumented training preservation was checked for all five SG configurations, including replay contents and available final network parameters. Snapshot serialization/replay was checked for weights, dynamic matrix and combined-controller decisions. An independent linear-plant test checks the common-input forecast comparison.

The original-network CUDA CLI checks saved all requested horizon (4) and combined (16) matched decisions with zero failed pairs. These use the compressed technical clock and are not production scientific results. An audit found no changes to any of the 124 pre-existing tracked scientific-code/data/workflow files, including the pre-existing user edit to the repetition wrapper. All 46 frozen original manuscript files still match their manifest.

The final regression run found one restored-helper clock issue; it was fixed in the new snapshot adapter. The targeted post-fix run passed 19 tests, and all 24 saved test snapshots reproduced their 48 original matched branches. The change report records the initial failure and repair.
