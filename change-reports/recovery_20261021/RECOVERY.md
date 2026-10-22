# NumPy, native E1 and remaining-run recovery — October 21, 2026

Amir explicitly requested repair of the NumPy crashes and E1 snapshot failure, recovery of every unfinished existing study, continued automatic combined launch, and faster E1 execution using free Aspen cases. Scientific budgets, seeds, schedules, learners and controller calculations are preserved. This record supersedes the earlier October 21 status for execution state; historical failures and artifacts remain available.

## Numerical recovery

The three saved Windows dumps identify an access violation in the NumPy 2.4.6 extension at the same module offset, 0xced76. The fault context is not a verified OpenBLAS-kernel defect. Both private and optional Conda BLAS libraries were loaded; this does not establish which library or mechanism caused memory corruption. No hardware diagnosis or BIOS/system modification follows from these records.

New workers use a separate Python virtual environment with the official NumPy 2.5.3 Windows CPython 3.13 wheel, SHA-256 `71cad2b2a7451ab79d8f5e71b453485b6775963d5cf794179144a7463fe6e8ec`. The private SciPy 1.17.1 build is retained. Private NumPy/SciPy BLAS use one thread and conservative Nehalem dispatch; NumPy X86_V3 dispatch is disabled. File inventories, package paths and bootstrap hashes are verified. The shared Python environment and processes already running were not modified. Package source/version: [official NumPy release](https://github.com/numpy/numpy/releases/tag/v2.5.3); exact wheel metadata and file manifest are retained in `runtime/`.

A 50-second production-kernel stress check passed 2,911 Markov L-BFGS fits and 8,733 linear-algebra batches with SLSQP/PyTorch coexistence. Its fitted Markov result matches the previous reference exactly. These bounded checks and current worker liveness do not establish multi-day stability or prove the underlying native-crash cause. Optional dependencies may still load Conda BLAS.

Only the pending combined cohort's interpreter/runtime override was changed before handoff. Its sealed scientific source, model, assignment and original bootstrap hashes remain intact. The existing coordinator subsequently started all ten combined jobs automatically; individual `isolated_runtime.json` files report private NumPy 2.5.3.

## E1 restoration

The original LoadSnapshot plus zero-time integrator initialization changed composition by approximately 1.75e-6 and temperature by 0.00305. Testing the last integrated feed did not remove the drift; a pending-feed explanation was not confirmed. Opening a saved project without restoring its snapshot starts at time zero and is not a continuation mechanism.

The process-local repair retains LoadSnapshot, physical specifications and the zero-time initialization, then uses Aspen's documented Results.CopyValues with every native-variable class selected. It verifies the original, tighter immediate-restore bounds again after copying. No fidelity tolerance is loosened. Cached Python plant/controller state is restored after this operation; counterfactuals continue on a separate application and never rewind the training application.

At the actual failed production snapshot (step 5202), three repeated four-step continuations passed. Post-copy diagnostic differences were zero, advancing time was correct and repeated traces agreed. A separate real four-channel E1 learner integration produced four completed technical matched decisions; a fresh process restored the full step-80 checkpoint and advanced to step 88 with complete Python-state hash validation. These are technical checks, not completed production E1 evidence.

## Full checkpoint and startup repairs

New recovery repetitions and E1 save the native Aspen state plus learners, optimizers, replay/n-step state, RNG, observer/controller state and histories. A private copy is saved, so checkpoint export cannot change a canonical source case reserved by the queue. Fresh-process residual recovery passed step 8 to step 16 with a complete Python-state hash. The native loader preserves input/output array identities used by controller aliases. JSON checkpoint metadata is converted to JSON-safe types before the existing Windows atomic-replacement retry.

The first Markov launches exposed two checkpoint-adapter contracts: its range(nFE) loop is inside a try/finally, and its cycle length is stored inside ctx. Both were stopped before cycle 1. A separate process-local compiler preserves the original statement order, tail, helper semantics and cleanup, supports that exact try/finally structure, and exposes the unchanged cycle length for checkpoint accounting. Unsupported structures fail explicitly. Historical startup failures are preserved. Real Markov step-8 save and fresh-process step-16 continuation passed before the final production relaunch; their evidence is `markov_native_8.log`, `markov_native_16_corrected.log` and the technical markers in the recovery attempt.

An initial bounded restart harness omitted the restored-step continuation length and therefore did not stop at step 16. Only that technical Python process and its unique case-3 Aspen tree were stopped after executable/window/case, ancestry and creation-time verification. Its step-8 checkpoint checksum remained intact and all production workers were preserved. The corrected harness explicitly advances eight additional steps and verifies a new step-16 index and complete restored-state hash. The Markov companion records explicit Aspen process ownership in production as well.

Threshold recovery retains all four populated, finite policy exports and now also saves complete training state, unlike the older inference-only exports. It restarts the unchanged 200-cycle seed-7 experiment from its original initial condition. New weights belong to the new trajectory. The old failed trajectory and partial inference weights are preserved separately.

The focused suite passes 38 tests, including unchanged E1 tests, drift rejection, zero-time checks, branch isolation, policy/full-state round trips, controller aliasing, NumPy JSON metadata, original Markov compilation/cleanup/restart ordering, and virtual-environment worker PID/progress reconciliation. Real native tests provide separate evidence beyond those tests.

## Parallel execution and visibility

Five existing E1 configurations can now run concurrently: combined on canonical case 16, horizon 13, weights 14, Markov 15 and residual 17. Case 17 is an isolated, byte-identical source copy; each E1 worker owns separate main/branch working copies under its canonical lease. The one-seed 300-cycle, 3,200-pair budget is unchanged. Markov SG seed 23 uses case 3; residual plain seed 23 uses case 11. Combined repetitions retain cases 1,2,12,4–10. There is no recovery/queue case overlap.

Supervisors run through independent interactive Windows Task Scheduler brokers, with no recurring triggers or execution time limit. Visible recovery consoles and a 30-second read-only all-run progress viewer are started. Virtual-environment redirector PIDs are reconciled to the actual interpreter child. Closing the viewer does not stop simulations. Verified full checkpoints permit bounded automatic retries for native worker crashes; structural errors stop for inspection.

The final timestamp, per-worker state/PIDs/cycles, checkpoint indices, owned Aspen identities, source guards, memory, combined handoff and protected-worker completions are in `recovery_assessment.json`. Earlier `launch_assessment.json` intentionally retains the startup issues discovered during validation. Finished polymer/E1/E2/E3 evidence is retained. All four workers healthy at the preceding audit finished naturally; the original combined coordinator retains its creation identity.

## Time and evidence limits

Comparable complete column repetitions took 30.8–35.9 hours before the increased concurrency; allow approximately 36–60 hours for the new combined cohort and remaining repetitions, conditional on throughput and no further failures. Threshold recovery previously reached 129/200 cycles over several hours; an 8–12-hour planning allowance is provisional.

E1 remains the longest task. Its unchanged budget includes approximately 600,000 main and 1,664,000 counterfactual advances. Parallelizing five configurations removes their sequential wait; the largest combined configuration still requires approximately 952,000 advances. At one second per advance that is about 11 days before transfer/checkpoint overhead. Current full-load branch throughput is not established, so roughly one to several weeks remains a conditional planning range, not a completion promise. No sampling budget was silently reduced.

## Preservation and files

Scientific runner/controller sources, shared numerical environment, original results, notebooks and canonical source models are unchanged. The unrelated pre-existing `run_polymer_phase2_repeats.py` edit is excluded from this task's commit. No TeX or PDF is changed; all 46 frozen original manuscript hashes verify.

Inspected: crash dumps and module context; original numerical/bootstrap manifests; installed Aspen CopyValues documentation; native reproduction/integration/restart evidence; authoritative runner/checkpoint code; all current manifests, statuses and logs; canonical-case assignments and process ownership; completed artifact/provenance records; queue source/model/bootstrap guards; system resources; paper memory and original freeze hashes.

Changed: isolated numerical, snapshot, full-checkpoint and Markov adapters; separate recovery supervisors; the read-only progress reconciler; focused tests; this report and compact verification/provenance records. The paper repository receives a dated execution record, compact evidence and affected memory/decision updates. Changes are committed locally without pushing.
