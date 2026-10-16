# E3 polymer combined: native-crash recovery

Amir authorized fixing and restarting only failed E3 polymer combined, retaining
original controllers and other active jobs. The associated evidence review is
restricted to implemented E1--E3; new distillation extensions require discussion.

## What the records establish

Original attempt `Results/e3/20261015_042751_513bff/polymer_combined_s7/a1`
completed its last logged cycle 99, saved 100 release-window prediction origins,
and exited with 3221225477 (Windows access violation, `0xc0000005`). Faulthandler
places the fault in `prediction_improvement_score`, called by SciPy's numerical
gradient calculation for the original L-BFGS-B least-squares correction fit.
The process died before final training export. This is not an optimizer-returned
failure status or a Python exception that the worker could catch and serialize.
The root manifest correctly classified the attempt as failed, although its
interrupted worker-status file still says running. PID 10108 was no longer alive.

The old trace does not identify an exact failing binary instruction or reproduce
the cycle-100 inputs. No complete optimizer/replay/plant checkpoint exists for
resuming at that point. The immediate native failure is established; the
underlying binary or hardware trigger remains unproven. Earlier E1/E2 native
failures in NumPy/OpenBLAS are documented separately, and their past package
replacement did not prevent this E3 failure. Do not claim a verified specific
NumPy-version defect or guaranteed permanent cure.

## Isolated runtime mitigation

`run_polymer_e3_recovery.py` is a new bootstrap outside every active source
fingerprint. It uses the existing E3 scheduler, worker, training entrypoint,
capture and analysis implementations. None of those files is edited.

Only the restarted child receives `NPY_DISABLE_CPU_FEATURES=X86_V3,AVX2,FMA3`
and `OPENBLAS_CORETYPE=NEHALEM`, with the existing single-thread numerical settings.
The worker checks the actual backend before training: NumPy's grouped X86_V3
dispatch is disabled and OpenBLAS reports single-threaded Nehalem kernels. This
does not disable every instruction in every loaded library. The saved installed
versions remain Python 3.13.2, NumPy 2.4.6, SciPy 1.17.1 and CUDA PyTorch 2.13.0.
No installation, shared environment variable, package file or OS setting changes.

The use of process-start CPU-dispatch controls follows
[NumPy's crash troubleshooting](https://numpy.org/doc/stable/user/troubleshooting-importerror.html#segfaults-or-crashes)
and [OpenBLAS runtime-variable documentation](https://www.openmathlib.org/OpenBLAS/docs/runtime_variables/).
These sources support the available mitigation controls; they do not establish
the original crash's exact cause.

Each worker records the actual runtime and bootstrap hash in
`isolated_runtime.json`; `recovery_launch.json` records the real command and
runtime overrides. These join the original worker's final artifact inventory.
The source fingerprint remains strict for original implementation files.

## Preservation and restart

The bootstrap verifies every completed nominal/matrix artifact using the
existing E3 checks, refuses a live prior combined PID, and locks only the polymer
E3 study. The original scheduler skips both completed stages and reserves `a2`
for combined. All original options remain unchanged: seed 7, 300 cycles,
original warm-up/critic-only hand-off/release, Scenario 2 profile, original
networks/rewards/constraints/bounds/margins, and 100 prediction origins in each
of four windows with lead times 1, 4 and 8. Training restarts at cycle 1 because
there is no exact training checkpoint. After completion the existing E3 analysis
runs normally.

The original manifest is copied to the study's `recovery/<UTC timestamp>/`
record before the scheduler adds an attempt. All 105 files in failed `a1` are
hash-sealed and preserved, and no nominal or matrix job is retrained. Other
E1/E2, E3 distillation, and distillation-repeat source inventories are checked
before and after repair; no stop/kill signal or source edit targets those jobs.

The authorized restart is already dispatched. Parent PID 38588 launched worker
49624 in `polymer_combined_s7/a2`; the warm-up log advances. This is a restarted
full study, not a completed result. No new distillation study is launched.

## Verification and records

43 regression/contract tests passed, including refusing duplicate/live workers,
preserving completed stages, scope restrictions, source-change rejection,
configuration preservation, and original E3 scheduler contracts. Seven expensive
unchanged capture/analysis tests were excluded. Separate native probes executed
2,000 original LS-score calls and 100 original bounded fits under each runtime,
using captured plant matrices and deterministic synthetic histories. Recorded
scores and fitted coefficients agree exactly on those fixtures. Both probes
passed; the short default probe did not reproduce the original long-run crash.

Records are under `change-reports/e3_polymer_recovery_20261015/`: the numerical
probe source/results/comparison, regression log, old failed-file hashes,
protected source inventories, restart-plan log and production launch metadata.
Generated long-running worker output remains in its original E3 study tree.

Files inspected include the E3 scheduler/runtime/capture/prediction/analysis,
the original Markov LS score/fitter and combined call site, NumPy/OpenBLAS runtime
introspection, prior native-crash records, process inventory, original run
manifests/logs and existing E3 tests. Changed no existing implementation file.
Added only the recovery bootstrap, focused tests, native probe, small records
and this report. The pre-existing dirty polymer repetition wrapper is excluded
from the commit. No TeX change or compilation is part of this task.
