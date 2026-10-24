# Combined ETA and free Aspen capacity — October 24, 2026

At 14:48 Toronto, combined is the sole unfinished E1 configuration: 192/320 matched comparisons and a complete cycle-204 checkpoint. Its accepted PID 15940/creation 134373287822018420 is unchanged and progressing. The full campaign has 512/640 comparisons; 128 remain. There is no current failure. Horizon, weights, Markov and residual are complete at 300 cycles/80 pairs each; repetitions and their combined coordinator remain completed.

Markov finished at 14:36:12 Toronto and residual at 14:37:57. Their new immutable `completion.json` receipts independently verify both final trajectory hashes, final matched Python/native checkpoint hashes and source maps, all 160 trace CRC/SHA-256 checks, completed four-window sampling, final analysis summaries, all 143 source seals, retired worker generations and no remaining owned native applications. Each has zero failed pairs. Markov's later supervisor diagnostic refers to the superseded original combined identity after Markov's scientific completion; the completed manifest/output are intact. Residual has no such diagnostic. Neither needs recovery.

## Estimated remaining wall time

The estimate separates main training from E1 diagnostics. Four combined learners advance together along one trajectory. At a sampled decision the existing runner suspends that trajectory, runs the policy branch and then the supervisor branch on a separate application, saves their paired evidence, and resumes training. Each branch uses the same decision-time controller/learner state and RNG state. The last recorded decision in the timing assessment is step 81582; the subsequent complete training checkpoint reached 81600. This difference is immaterial to the rounded estimate.

The exact remaining prospective schedule contains 128 comparisons, including 14 extended pairs. It therefore requires 68,000 additional diagnostic control steps (two 200-step or two 800-step branches per pair), plus at most 38,417 main training steps at the assessment cutoff. Already saved/replayed versions are not additional independent samples.

The model uses 17 intervals between newly closed traces in the current accepted combined generation, after 15:26 UTC; the restart/replay discontinuity is excluded. Sixteen ordinary intervals identify training seconds per step and the regular comparison cost. One extended interval identifies the additional rollout cost. The ordinary-pair fit has a 25.4-second RMS residual. Its fitted coefficients are 1.079 seconds per main training step, 0.959 seconds per branch step, and 167.4 seconds of fixed comparison overhead. Raw timing rows and the exact remaining schedule counts are in `assessment.json`.

$$ T_{\mathrm{seconds}} \approx 1.079 C + 0.959 B + 167.4 N. $$

Here C is remaining main control steps, B is remaining diagnostic branch control steps, and N is remaining matched comparisons.

| Component | Estimated remaining hours |
| --- | ---: |
| Coupled main training | 11.5 |
| Diagnostic branch rollouts | 18.1 |
| Native restoration and other comparison overhead | 6.0 |
| Current serial execution | 35.6 |

Use **30–40 more hours as a planning range**, assuming no further failures: approximately Sunday October 25 at 20:45 through Monday October 26 at 06:45 Toronto, centered near Monday 02:20. This is a conditional engineering estimate, not a statistical confidence interval. Main-step cost is extrapolated from short within-window gaps into the long no-capture training interval; only one extended-pair interval is available, future plant/solver costs can change, and Markov/residual retirement may improve resource availability. Future failures, retrofit time and checkpoint suffix replay are excluded. Ordinary pairs currently take about 9–11 minutes; the observed extended pair took 28.5 minutes. A simple remaining-pairs/rate calculation would omit much of the remaining main training.

## Can free Aspen cases help?

**There is available capacity, but the active runner cannot attach extra workers live.** Existing canonical source model/job hashes verify for cases 15 and 17. Their wrapper leases were acquired together and immediately released without opening Aspen. The only live native applications belong to combined. At the capacity probe the machine had 24 logical processors, about 3.5% instantaneous CPU load, 63.7 GiB physical RAM and 56.9 GiB commitment headroom. These are point-in-time capacity/ownership checks, not a native parallel performance benchmark. No simulator or additional training worker was launched.

The suitable split is to run the policy and supervisor counterfactual branches in two separate owned processes, retaining the same frozen decision-time networks, controller/normalizer state, native snapshot and RNG for both. Keep the advancing four-learner training trajectory together. The current frozen-agent implementation shares read-only training network parameters while training is suspended; ordinary threads would also share process-global RNG/restoration/patch state. A valid process pool needs independent serialized frozen copies and native handles, the existing restoration tolerances and original trace/learner integrity checks. Duplicating training or assigning its four interacting learners to independent trajectories would change the experiment.

If the branch computation halves, the model projects **26.5 hours** with fixed overhead unchanged, or **23.5 hours** if all comparison overhead also halves. These are idealized capacity scenarios, **not measured speedups**; process startup, serialization, validation, checkpoint migration and any replayed suffix can reduce the saving. More cases alone cannot remove the roughly 11.5-hour serial training estimate. An asynchronous multi-decision queue would require additional frozen-state, checkpoint, ordering, pinning and recovery work and is not implemented by the active source.

There is no existing live add-worker switch. Activating this split would require a new isolated parallel comparison adapter, testing against saved serial raw traces on free cases, refreshing the complete backup/recovery plan, and a guarded combined-only checkpoint migration. The earlier memory-only restart approval does not by itself establish that this unimplemented adapter has passed native equivalence tests. No retrofit was activated during this assessment. **Recommendation for this final configuration: continue the accepted progressing worker and budget 30–40 hours.** The projected 9–12-hour saving excludes adapter development, native validation, migration and replay costs. A parallel implementation can be assessed against these concrete baseline timings and acceptance requirements.

## Preservation and verification

The final collection `../hourly_run_checks_20261024/audit_20261024T184833.json` has no observation needing diagnosis; ten newly saved/replayed closed traces and all 143 source seals verify. Retention is active and verifies the latest two complete pairs for every configuration: combined 81600/81200 and each completed standalone 120000/119600. Headroom is 56.9 GiB, disk free space 335.4 GiB and every live native app is combined-owned. Original results, pinned dependencies and retained lossless archives remain intact. No live simulator call, process signal, training launch/restart, source/queue/package/paging change, deletion, administrator access or OneDrive reopening occurred. The existing viewer registration remains current and hourly monitoring continues for combined alone.

Inspected: `RUN_RECOVERY_STATUS.md`; current accepted recovery receipts, plans and job source models; `run_column_e1.py`, `run_column_e1_budget.py`, `column_e1/runtime.py`, `column_e1/checkpoints.py`, `column_e1/replay.py`, `column_e1_budget.py`, `column_e1_process_recovery.py`; `review_experiments/e1/study.py`, `runtime.py`, `snapshots.py`, `analysis.py`; immutable prior hourly audits; stable current status/checkpoint/sampling/ownership aliases; append-only captures and closed trace timestamps; final trajectory/checkpoint/analysis files; actual process identities, case leases, retention and resources.

Changed: this report's standard-library read-only `assess.py`, `assessment.json`, `free_cases.json` and this status; new Markov/residual immutable completion receipts and concise statuses; root/hourly statuses and final audit. Verification: completed-file hashes/CRC, full source seals, prospective schedule reconciliation, explicit separation of measured timing and unbenchmarked projections, JSON/Python parsing and scoped whitespace checks. No numerical simulator test was run for a feature that was not implemented. The unrelated dirty `run_polymer_phase2_repeats.py` and manuscript remain untouched.
