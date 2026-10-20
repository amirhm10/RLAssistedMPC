# All current studies — October 20, 2026, 12:59 Toronto

Audit timestamp: `2026-10-20T16:59:26.963393+00:00`. There are **22 unfinished production jobs: 12 running and 10 queued**, plus **one failed/held technical snapshot pilot**. Full column E1 and the separately planned validation/sensitivity work are outside this existing queue. No new current worker failure, unregistered experiment worker or completed-artifact verification failure was found. Historical crashed/superseded attempts remain preserved.

## Finished evidence

- Polymer repetitions: **50/50**; saved successful worker records, 300-cycle counts and result-file presence verify.
- Column repetitions: **29/50**; all horizon/weights runs and nine Markov seeds. Successful 300-cycle worker records, cleanup, metadata and result hashes verify.
- Polymer E1: **3,200 unique completed matched pairs**, comprising 1,600 standalone plus 1,600 combined pairs. Source hashes and zero failed pair records verify; the merged analysis is complete. The old residual run's full training trajectory export is absent, but its 400 matched pairs remain validated and retained. No repeat of those pairs is needed merely because the old export guard failed.
- Polymer E2: **five methods plus the offset-free baseline** have complete, finite 160,000-step trajectories: LS, BO, threshold, SG and repaired regularized ARX. Shared postprocessing contains all six sources and 30 metric rows. All registered trajectory hashes verify. The ARX local `postprocess_status.json` says waiting from an earlier finalizer invocation; the authoritative shared `analysis/e2/provenance.json` is complete and contains the verified ARX/SG sources.
- Polymer E3: **3/3** stages; full saved artifact inventories verify.
- Column E3: **3/3** stages; full saved artifact inventories verify.
- Column E2 comparisons: **SG, threshold and LS complete**, each with 200 cycles/80,000 steps and verified final artifact inventories/cleanup.

## Existing work still running or queued

- **One Markov SG repetition**, seed 23 on case 3: live, cycle **108/300** at the audit. This is the isolated runtime retry; the old cycle-62 native failure is historical.
- **Ten residual repetitions**: all worker PIDs are live and have reached cycle **2/300**. SG seed 23 uses case 12; the other seeds use 1, 2, 4--10.
- **One column BO comparator**: live; **35/40** valid calibration trials saved. The current log cycle is inside calibration. The final **200-cycle evaluation follows calibration** and is not complete.
- **Ten combined repetitions**: queued after residual through the live independent coordinator. SG seed 23 will also use case 12. The coordinator is live and not waiting for the ongoing Markov case-3 retry.

The 12 production PIDs agree with an independent Windows process inventory. No additional experiment worker was found. Recent active-worker log tails contain no native-fatal/Python traceback, and all queried worker lifetimes are live. This is a current health check, not a guarantee for the rest of the long runs.

## Known held work and export limitations

1. **Column snapshot pilot, case 16: failed and held.** Native snapshot restoration has not demonstrated advancing time/repeatable continuation. Full column E1 remains blocked. Repair this through bounded isolated restoration/clock tests before another production prefix or full E1 run.
2. **Old polymer threshold inference export: empty.** Loading the saved checkpoint finds zero tensor elements. Its completed trajectory remains usable for descriptive comparisons, but the policy cannot be restored for frozen-policy tests. Recover a usable existing policy export, or scope a targeted rerun if no weights survived; do not silently represent the empty export as a trained policy.
3. **Historical ARX/postprocess and E1 export labels:** retain their provenance. The repaired ARX/shared E2 evidence is complete, and residual E1 pair captures are usable. Those labels do not add current production reruns to the 22-job total.

## Recommended order

Keep the 12 existing workers running and let the ten combined jobs start through the existing queue. Retain completed polymer/E1/E2/E3 data and avoid duplicate training. Independently resolve the case-16 native restoration problem and recover the threshold inference export. As the column results complete, validate saved bundles/checkpoints and assemble the paired seed statistics and E2 comparison analysis. Frozen-policy unseen-reference evaluations, full column E1 after a validated adapter, and E4/E5 work remain separately scoped unlaunched work; this audit starts none of them. E6/E7 retain the accepted justification/limitations route without new benchmark or architecture campaigns.

Evidence: `all_runs_20261020_165926.json`; reproducible read-only audit: `audit_all_runs.py`. The audit reads status/manifests, successful result records, registered artifact hashes, E1 captures, E2 primary-array shape/finiteness and shared analyses, E3 inventories, BO trial records, worker log tails and the Windows process inventory. It does not open Aspen, stop/restart a job, modify scientific sources or regenerate result metrics/figures. Changed files are this audit/report/evidence and the saved overview/workflow records. The unrelated pre-existing edit to `run_polymer_phase2_repeats.py` remains preserved.
