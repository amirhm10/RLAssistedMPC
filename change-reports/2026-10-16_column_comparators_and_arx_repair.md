# Column alternatives, polymer ARX diagnosis and E1--E3 audit

Amir authorized developing the discussed column alternatives, diagnosing/fixing
the polymer ARX failure, checking current E1--E3 runs, and exposing progress.
New production studies are prepared for later launch. Existing scientific
processes continue; no controller, simulator case, installed package or old raw
result was changed. The pre-existing dirty `run_polymer_phase2_repeats.py` remains
untouched and outside this commit.

## Implementation

- New `run_distillation_comparisons.py/.cmd` and `revision_comparisons/column.py`:
  threshold/SG, LS and BO using original 200-cycle column entrypoints, one seed,
  unchanged warm-up/hand-off/evaluation, case locks and numbered recovery attempts.
- BO uses complete fresh initialization and replayed nominal prefixes to avoid
  inventing Aspen snapshot restoration. 40 trials require 192,000 calibration
  control steps in addition to 80,000 main steps. Trial/worker timing is separate.
- The saved column baseline is re-scored in sidecars with the common current
  reward after schedule/scaling/error-alignment checks. Its cached old reward and
  original trajectory are preserved. No new nominal production run is needed.
- New `revision_comparisons/arx.py` and `run_polymer_arx_revision.py`: persistent
  nominal-prior RLS plus explicit gain admission. Historical ARX code/results are
  unchanged. Repaired production and column ARX are not claimed completed.
- Standard-library progress viewers query CPU/exit status without signaling
  processes. Visible CMD viewers show the already-live E3 recovery and a summary
  of cycles/decisions/CPU. Closing a viewer does not stop the scientific study.
- `audit_revision_studies.py` records current states, hashes and source freezes.

See `DISTILLATION_SCENARIO1_COMPARISONS.md` for commands, mathematical definitions,
reference links, units, provenance and limits.

## ARX diagnosis

Historical trajectory SHA:
`39dd0c8427eed56464cb6b48e7643c04f0cd26d5946cae913024fe62569676ab`.
All admitted poles are stable, but 8775 admitted control steps have a reversed
monomer-to-temperature DC gain. First occurrence is step 9200 (cycle 12). The
final gain is -0.250844 versus nominal +0.019425. Final temperature is about 444 K
and both inputs saturate. There is one unsuccessful optimizer call at index
80801, insufficient to attribute the behavior to pervasive solver failures.

Replaying the original RLS equations reproduces the saved coefficients within
2.834e-10. Covariance diagonals reach 1e8 in both output models. The old nominal
prior fades while poorly excited directions lose information. The revised
information-form estimator maintains its nominal prior, with minimum information
eigenvalues about .001; model admission blocks reversed/excessive DC gains and
retains the last admitted predictor. This fixes the identified admission gap;
new closed-loop performance is not inferred from offline replay.

Derived record: `column_comparisons_20261015/arx_diagnostic_a2/arx_diagnosis.json`.
It includes original-source SHA, per-cycle measurements, exact algorithm replay,
repaired-model rejection counts and diagnostic scope.

## Run audit and why E1 looks paused

At 2026-10-16 04:13 UTC, artifact/source audit confirms:

- Polymer repeats: 50/50 completed.
- Column repeats: 10 completed, 10 weights running, 30 queued; no current failed job.
- Polymer E1: 400 valid completed pairs in each standalone channel; combined is
  live in its separate recovery. Residual lacks its historical main training
  export after a source-guard failure, although its 400 matched pairs survive.
- Polymer E2: LS/ARX/BO/threshold full trajectories and stored SHAs verify. ARX is
  computationally completed but scientifically deteriorated. SG remains queued
  behind E1 combined after the old final-export failure.
- Polymer E3: nominal/matrix artifacts and full file inventories verify; combined
  a2 is live after a1's native access violation. No new E3 error is observed.
- Column E3: nominal full artifact inventory verifies, matrix is live, combined
  is queued. The root's old pending label does not supersede live job status.

The user saw cycle 191 / 460 matched decisions. Direct checks then found 470,
473 and additional completed decisions with increasing CPU time. This is active
matched counterfactual work inside a cycle, not an E2 worker hung at cycle 191.
E1 combined targets 1600 pairs, with 200-step branches and selected 800-step
extensions. No sampling budget, horizon or training procedure was reduced.

The audit verifies unchanged E3 inventories (103 files each), E1 recovery (97),
column repeats (86), and the original 46-file manuscript freeze. Earlier failed
attempts remain visible and are not relabeled as completed results. Final E1
combined/E2 SG/E3 analyses await their actual exports.

The final 04:22 UTC snapshot confirms **489/1600** completed combined pairs,
training cycle 192 and increasing CPU time; E3 polymer combined a2 reaches cycle
60, and E3 column matrix reaches 213 with 300 captured origins. Current repeat
counts remain 50/50 polymer and 10 completed/10 running/30 queued column. Record:
`column_comparisons_20261015/final_study_audit.json`.

## Validation

- `tests/test_revision_comparisons.py` plus existing comparator/E3/review recovery
  tests: **51 passed**, four expected GP kernel-bound warnings.
- Original column entrypoints on a compressed technical clock with an identified
  model stand-in: nominal, LS, BO, threshold and SG all pass; all original neural
  architectures and four-agent inference exports are exercised, and all 11 owned
  fake cases close. No live Aspen execution. Record:
  `column_comparisons_20261015/offline_column_checks_a2/summary.json`.
- Repaired ARX: 800-step technical smoke, zero failed solver calls; preserve its
  explicitly technical status and runtime metadata. No production claim.
- 160,000-observation original/repaired offline RLS replay completed.

## Files inspected and changed

Inspected: both repository instructions; focused research/testing/Aspen skills;
original column Markov/combined/baseline setup, distillation scenarios/config/data
I/O/plant; shared baseline/Markov/combined/MPC/threshold/E1 instrumentation; original
comparison methods/provenance/checkpoints/tests; active E1/E2/E3/repeat manifests,
logs and saved artifacts; original manuscript freeze and current paper memory.

Changed only new files: the column and ARX wrappers, `revision_comparisons/`
package, progress scripts, audit script, focused tests, this note and the method
documentation/evidence listed above. No `.tex` source or PDF is edited, and the
user's document editor remains open. The corresponding paper record will remain
in `manuscript_revised/`, with scoped workflow synchronization. Local commits only.
