# Isolated OpenBLAS mitigation and seed 23 queue recovery

Author request: repair the OpenBLAS crash, restart failed Markov SG seed 23 without interrupting other runs, and keep the existing remaining queue able to start.

## Observations and change

The October 19 03:59 Toronto Windows Application fault event names Conda OpenBLAS 0.3.33, exception `0xC0000005`, offset `0x276a6`, in worker 9304. The fatal Python stack is the production Markov prediction-improvement objective inside L-BFGS-B. The run ended at cycle 62. Its worker status remained stale `running`; the original scheduler would stop before the next category when the Markov group finished. The previous single-threaded Nehalem/disabled-X86_V3 mitigation did not prevent this crash. The exact underlying native defect is not established.

`review_blas_runtime.py` selects a private, hash-pinned official wheel overlay before scientific imports. NumPy/SciPy keep versions 2.4.6/1.17.1 but use the wheels' private OpenBLAS builds 0.3.31.188.0/0.3.30 for their math. Installed packages/DLLs are untouched. All 2,465 private files are verified before each new worker. Actual paths, dispatch, thread counts and bootstrap hashes are recorded with the attempt and copied alongside a completed result. Optional dependencies can still load Conda's BLAS; this change specifically replaces the NumPy/SciPy binding implicated in the crash.

`recover_markov_seed23.py` creates separate retry/residual/combined group manifests and an aggregate 50-job manifest. It never writes the live old supervisor's manifest. It starts only seed 23 now, from original initial conditions, and waits for all old workers, their supervisor and the retry to finish before resource-checked queue handoff. Residual then combined can each start even if a previous group failed, after confirmed cleanup. A failed run remains failed. Unverified ownership, changed scientific sources or ambiguous launches stop handoff visibly rather than risk duplication. The coordinator and viewers are launched by the interactive Windows scheduler outside Codex's process tree, without recurring triggers or timeout.

Cleanup accepts only Aspen apps recorded by a dead worker, plus named descendants with verified creation identities. Termination rechecks creation on the same process handle. At actual launch the old app was already absent; the cleanup audit records no termination actions. Case 3 is now owned by the new worker. All nine protected Markov workers retained their PIDs and creation times.

The fake-worker integration reproduced a Windows `PermissionError` replacing a manifest while readers briefly held it open. New recovery processes use unique temporary files and a bounded five-second retry for these atomic writes. The shared scientific scheduler file and live process code were not changed.

## Validation

- 39 focused tests passed: `test_markov_seed23_recovery.py`, `test_recover_review_runs.py`, `test_distillation_phase2_repeats.py`. The integration launches 21 fake workers through the real scheduler and verifies 20 retained results are skipped, all 20 queued jobs dispatch, combined runs still dispatch after injected retry/residual failures, and old manifests/results remain intact. Other tests cover leases, duplicate refusal, source/hash guards, ownership and PID reuse, and transient atomic-write locks.
- A 40-second private runtime stress exercised the actual Markov LS implementation 2,738 times, 8,214 linear batches, contiguous/Fortran/strided GEMV, NumPy solves, SLSQP and PyTorch gradients. Maximum linear residual was 1.84e-14. Fitted coordinates matched the reference build exactly for this synthetic fixture. This does not reproduce or exclude a multi-day native crash, nor validate the physical plant.
- Official PyPI metadata verifies both wheel SHA-256 values. See `wheel_provenance.json`; no global install or downgrade occurred.
- Launch evidence verifies new worker 28652, private runtime, owned Aspen identity, independent scheduler parent/session, nine unchanged protected workers, 20 retained result hashes, and 20 queued jobs. The original scientific inventories remain unchanged: repeat 86, E1 97, both E3 inventories 103 files each. All 46 frozen manuscript files remain unchanged. The shared OpenBLAS DLL matches its original Conda package hash.

Evidence: `recovery_assessment.json`, `private_stress.json`, `reference.json`, `windows_fault_event.json`; production `Distillation/Results/review_recovery/20261020_openblas/`. These are startup/repair records, not completed seed-replication evidence. The separate snapshot pilot is still held; its failure is outside this repair.

To reconstruct the local ignored runtime, use the recorded environment's Python to download only the two wheels from `https://pypi.org/simple` with `pip download --only-binary=:all: --no-deps --dest change-reports/run_recovery_20261020/runtime/wheels numpy==2.4.6 scipy==1.17.1`, then run `python -B review_blas_runtime.py --prepare`. It checks the pinned hashes and refuses to overwrite differing files. Do not reinstall or replace packages in the shared environment while experiments run.

## Files inspected and changed

Inspected: original repeat manifests, failed seed 23 logs/status/job/ownership/runtime, Windows fault event, package metadata/DLLs, active worker identities, `utils/markov_runner.py`, `run_distillation_phase2_repeats.py`, `recover_review_runs.py`, `review_restart_recovery.py`, `column_review/common.py`, original runtime diagnostic, progress viewer, focused tests and paper freeze manifest. Verified scientific sources are not edited.

Changed/added: new runtime/bootstrap and coordinator, `revision_comparisons/progress.py`, `tests/test_markov_seed23_recovery.py`, this dated evidence/diagnostic package, and `RUN_RECOVERY_STATUS.md`. Paper workflow records are synchronized separately. The unrelated existing modification to `run_polymer_phase2_repeats.py` is preserved and excluded from the commit. The runtime wheels and extracted packages are local ignored artifacts, not committed binaries.
