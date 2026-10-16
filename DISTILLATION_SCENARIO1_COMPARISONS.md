# Distillation Scenario 1 comparators and polymer ARX repair

Author-approved development, 2026-10-16. The new column alternatives are
threshold-supervised combined RL, LS-only MPC and Bayesian MPC-weight tuning.
A matching combined SG reference is included when a verified reference is not
supplied. One seed (7), the original column Scenario 1, and the original
controllers are retained. No new production study was started during development.

## Commands in CMD using rl-assisted-env

```cmd
call "C:\ProgramData\anaconda3\Scripts\activate.bat" rl-assisted-env
cd /d "C:\Users\hamediaa\OneDrive - McMaster University\PythonProjects\RL_assisted_MPC"
python run_distillation_comparisons.py --dry-run
```

The default is a read-only plan. After the selected Aspen case is free:

```cmd
run_distillation_comparisons.cmd --run --aspen-number 1
```

This runs threshold, SG, LS and BO sequentially. Existing column repetitions own
cases 1--10, and E3 owns case 11. The runner uses their same OS file locks and
refuses an occupied case. It does not close their processes or create model
copies. `--aspen-root` selects another directory containing an existing numbered
case. Aspen initialization and cleanup use the existing verified COM wrapper.

Resume a future comparison batch with:

```cmd
run_distillation_comparisons.cmd --resume "FULL_STUDY_DIRECTORY" --run
```

Completed methods are reused only after their job provenance and full file
inventory verify. Unfinished methods get separate numbered attempts, restarting
that method from its original initial condition. There is no exact training or
BO-trial checkpoint resume. A live previous worker prevents a duplicate launch.
`--sg-reference "COMPLETED_JOB_DIRECTORY"` accepts a reference produced by this
runner with matching seed, profile, dependencies and calibration specification.
Historical paper results are not silently certified as that reference.

The existing runs can be viewed without launching anything:

```cmd
show_review_progress.cmd
show_e3_polymer_progress.cmd
```

These are viewers. Closing a viewer leaves the scientific processes running.
The summary shows cycles, process CPU time, matched-decision counts and predictor
capture counts. An unchanged cycle can contain many expensive E1 branch rollouts.

## Fixed column protocol

- `legacy_200`: 200 cycles, two 200-step setpoint holds, 80,000 main control steps.
- Original disturbance/fluctuation profile, sample interval, setpoints, initial
  conditions, identification/scaling assets, input bounds, MPC settings and reward.
- Original fixed MPC: prediction/control horizons 6/3 and Q/R diagonals [1,1].
- RL: warm-up cycles 1--10, critic-only hand-off 11--13, original eligibility
  release during cycle 14, learning through 199 and evaluation in 200. Inclusive
  native step-boundary behavior, four original networks, replay and exploration
  schedules, downstream action checks and executed-action replay are preserved.
- Analytical LS adaptation stays active during final evaluation. BO weights are
  fixed during its entire fresh main trajectory.

The implementation executes the unchanged column Markov/combined entrypoints and
shared runners. Overrides are confined to method selection, the requested
Scenario 1 profile/seed, output paths, the chosen Aspen case and the threshold's
acceptance rule. Existing controller files and `comparison_studies/` are unchanged.
New code is isolated under `revision_comparisons/`, outside current studies'
source inventories. Future runs fingerprint their own execution dependencies.

## Methods

**LS-only:** original accepted least-squares dynamic-matrix correction; RL
proposals are disabled and no neural agent is created. The existing fit,
regularization, drift, prediction and nominal-cost checks remain.

**Threshold:** original combined RL learning, with the acceptance decision based
on E = max over outputs of absolute physical tracking error divided by the
original acceptable tracking band. Accept an eligible finite proposal when
E <= kappa; otherwise execute the native supervisor. Calibrate
kappa = max(1, 1.25 times the 95th percentile of E) using before-action nominal
measurements from cycles 1--10, then freeze it. Readiness and hand-off eligibility
remain enforced. The column obtains its own calibration value. Original critics
still train and compute shadow scores; no critic-inference cost saving is claimed.

**BO:** Q = [alpha Q01, Q02], R = beta R0, with alpha and beta in [.75,2].
Use the existing GP expected-improvement implementation: 40 trials including the
nominal point and four corners. Score negative mean original reward over original
cycles 11--12. Each trial starts a new owned Aspen case, replays the same nominal
cycles 1--10, then carries forward both the physical plant and complete observer
state. Slice the original feed/reference arrays for the scoring interval.

The nominal prefix is replayed because Python copying cannot restore a COM
simulator state. Each trial therefore uses 4000 prefix plus 800 scored control
steps: **192,000 calibration steps** for 40 trials, followed by a fresh 80,000-step
main run. Startup/settling and COM overhead are included in recorded wall times.
BO calibration cost is reported separately from main-controller time. Calibration
cycles are not held-out evidence; final evaluation is not used for tuning. This
two-parameter comparator and the multi-channel RL architecture have different
adjustable parameter spaces.

## Existing baseline and output bundles

The saved `Distillation/Data/mpc_results_disturb_fluctuation.pickle` has the correct
200-cycle trajectory, references, disturbance, scaling, bounds and fixed horizons.
Its cached rewards use an older reward configuration. The wrapper verifies the
saved physical error/input alignment and re-scores that trajectory with the exact
active common reward. It saves `baseline_audit.json` and `baseline_metrics.json`
beside the new outputs. The original baseline file is not modified or re-simulated.

New production bundles belong to `Distillation/Results/comparisons/scenario1/`.
Each method stores job/status JSON, resolved defaults, original training trajectory,
metrics, timing, numerical runtime metadata and hashes. RL methods export validated
nonempty inference weights for all four agents. These are inference exports, not
optimizer/replay/simulator checkpoints. Raw trajectories are saved before final
integrity checks; failed attempts remain failed and cannot become references.

The batch writes a combined CSV only from verified completed attempts. Metrics
cover cycles 14--20, 14--200, 191--199, 196--200 and evaluation 200. Output errors
use y[k+1] minus reference[k]. Report composition error, temperature difference,
each input's movement, fixed reward and available joint tracking-band occupancy.
Temperature differences in kelvin equal differences in degrees Celsius; original
raw simulator values are preserved. Inter-step controller timings exclude plant
integration and the first setup interval; worker and BO calibration wall times
include their respective overhead.

## Polymer ARX: diagnosed defect and separate repair

The historical ARX run completed all 160,000 steps but has invalid control
performance. Its pole-stability admission rule accepted models with reversed
manipulated-input gains. The nominal scaled monomer-to-temperature DC gain is
+0.019425; the final admitted value is -0.250844, while all admitted autoregressive
poles remain inside the unit circle. Wrong gain directions appear from cycle 12;
late operation remains around 444 K with both inputs at their upper bounds.
One optimizer call failed, but the failure is not explained by widespread solver
exceptions. The original RLS replay reproduces the saved coefficient trace within
2.84e-10 and reaches its covariance cap of 1e8 in both output models.

The old fading prior and weakly excited closed-loop data permit large covariance
and parameter drift. This is a model-admission/identification problem; the nominal
ARX history and delay implementation also pass state-space equivalence tests.

`revision_comparisons/arx.py` implements separately versioned regularized RLS:

$$ H_t = \lambda H_{t-1} + \phi_t\phi_t^T + (1-\lambda)H_0. $$

$$ b_t = \lambda b_{t-1} + \phi_t\Delta y_t + (1-\lambda)H_0\theta_0, \qquad \widehat\theta_t = H_t^{-1}b_t. $$

Use the original forgetting .995 and nominal prior covariance 1000 I. The nominal
prior remains present, so the information matrix does not lose that lower bound.
Admit finite stable candidates only if their DC-gain directions match nominal and
their gain magnitudes remain within [.25,4] times nominal; otherwise retain the
last admitted predictor. These fixed engineering bounds were selected before a
repaired production run. Their closed-loop adequacy still needs that run. The
prior introduces intentional estimation bias; no unbiasedness or closed-loop
stability guarantee is asserted.

```cmd
python run_polymer_arx_revision.py
python run_polymer_arx_revision.py --diagnose --replay
python run_polymer_arx_revision.py --smoke
python run_polymer_arx_revision.py --run
```

The first prints the plan. Diagnosis only replays saved observations. Smoke is
800 compressed technical steps. `--run` explicitly starts a fresh original
200-cycle production comparison in a new directory. No repaired production run
has started. **Column ARX is not enabled:** the polymer's fixed delays/model
structure must not be copied into the different column predictor without a
separate identification check.

## Validation and evidence boundaries

- 51 targeted/recovery regression tests pass. GP-bound convergence warnings are
  retained in the test log; no failed check is hidden.
- Original column entrypoints, all five method paths, original four-agent
  architectures and inference exports passed a compressed offline integration
  check using the identified linear model as a stand-in plant. All 11 owned fake
  cases closed. No Aspen instance was opened, and no smoke performance is evidence.
- Both original and repaired identification equations were replayed over all
  160,000 historical observations; the repair rejects the identified gain defects.
- Repaired polymer ARX passes an 800-step technical smoke with zero solver failures.
- The current E1/E3/repeat source inventories and all 46 frozen manuscript files
  verify unchanged. New production validation remains to be run later.

One seed supplies this selected comparator case, with replication handled by the
separate repeat campaign. Finish existing E1/E3 captures and their final analyses.
The proposed column E1 matched-rollout extension still needs verified restoration
of complete Aspen branch state; no such new experiment has been launched here.

## Background references

- Åström and Wittenmark (1973), [On Self-Tuning Regulators](https://doi.org/10.1016/0005-1098(73)90073-3): foundation for online LS/self-tuning, not a proof for this MIMO comparator.
- Forssell and Ljung (1999), [Closed-loop identification revisited](https://doi.org/10.1016/S0005-1098(99)00022-9): closed-loop identification assumptions and interpretation.
- Sorourifar et al. (2021), [MPC tuning using constrained Bayesian optimization](https://doi.org/10.1016/j.ifacol.2021.08.249), and Hirt et al. (2024), [Learning MPC parameters using Bayesian optimization](https://doi.org/10.1016/j.ifacol.2024.08.426): related outer-loop tuning; this is the simple two-weight comparator already specified for polymer.
- Lakhani et al. (2022), [Threshold-supervised RL-assisted PID tuning](https://doi.org/10.20517/ces.2021.15): related error-based supervision; their architecture and guarantees do not transfer to this selector.

Evidence, artifact hashes and current run status:
`change-reports/2026-10-16_column_comparators_and_arx_repair.md` and
`change-reports/column_comparisons_20261015/verified_study_audit.json`.
