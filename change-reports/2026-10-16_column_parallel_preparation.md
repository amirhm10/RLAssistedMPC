# Case 12--16 parallel column preparation

Amir requested clarification of the original 11--13 episode phase and authorized
development/verification of the next column jobs on cases 12--16. He explicitly
reserved the start instruction for later. This task prepares code and commands;
it opens no new Aspen application and starts no scientific study.

The paper's Results paragraph and SI table call 11--13 the handoff phase and
specify critic learning with action freeze under MPC control. Actors remain
frozen and policy eligibility starts in cycle 14. The original inclusive
sample boundaries are preserved. The launcher now names 11--13 critic-only.

Mapping is fixed: SG 12, threshold 13, LS-only 14, BO 15, E1 native-snapshot pilot
16. All five model files exist with identical hashes. Cases 1--11 are excluded.
Prepared batch: `Distillation/Results/column_review/20261016_prepared_cases12_16`.
It has five pending jobs and zero attempts. No launch command was invoked.

The wrappers hold a canonical model lease for the whole worker, reuse it for
nested calibration contexts, record fresh DispatchEx ownership by PID/creation
time, validate cleanup, preserve prior attempts, and run independently. They
provide live CMD logs/heartbeats, numerical-runtime checks and stack diagnostics.
Invalid artifacts, unknown cleanup, live prior processes, changed sources or
models and occupied cases stop the affected launch without closing other jobs.
Repeated starts reuse only verified completed results; a failed method receives
a fresh attempt from its original initial condition, not a fabricated training
resume. A verified four-method table excludes the restoration pilot.

The E1 pilot's native methods were confirmed from registered Aspen metadata,
without an application connection. Future live execution exports a converged
native snapshot after 400 nominal MPC samples and restores three identical
24-step continuations. It checks time advancement, feed/actuators, six internal
tray profiles, both outputs and the Python observer copy. It preserves the
native snapshot independently of routine cleanup. This pilot is not full E1
critic validation; all-agent/MPC-history integration follows a successful live
pilot. Unseen-reference evaluation still follows policy export/audit.

Pre-launch corrections in previously prepared comparison infrastructure:

- Standalone SG now creates baseline-audit sidecars, which its completed-result
  contract already required. Previously its direct worker could finish training
  but fail final artifact validation.
- SG/threshold retain common reward-band parameters so joint-band metrics are
  included. No reward value or learning equation is changed.
- Completed comparison status explicitly records/requires 80,000 samples and
  200 cycles. The new viewer discovers pending/running case-12--16 jobs.

61 focused tests passed. Five underlying comparator methods passed a compressed
offline integration check using original entrypoints/networks and an identified
linear stand-in; all 12 owned fake cases closed. Native-snapshot unit tests use
a fake stateful backend and reject hidden-state and time-advancement failures.
Live restoration remains to be tested when Amir authorizes starting case 16.

All active original source inventories and the 46-file original manuscript
freeze verify unchanged. No original scientific controller, agent, plant or
training-profile file is edited. The existing user modification to
`run_polymer_phase2_repeats.py` remains outside this task. No TeX/PDF is changed.

Inspected: submitted Results/SI phase description; native phase scheduler and
combined learning/gate call sites; column defaults/plant/COM cleanup; E3 resources;
comparison workers, fake integration and tests; registered Aspen type library;
active source/manuscript inventories.

Added: `column_review/`, `run_distillation_remaining.py/.cmd`,
`tests/test_column_review.py`, `DISTILLATION_REVIEW_PARALLEL.md`, this report and
bounded verification records. Updated: isolated `revision_comparisons/column.py`,
its offline checks/progress viewer and `run_distillation_comparisons.py`.
Original submitted/revised TeX and ongoing simulation inputs remain untouched.

Verification: `column_parallel_20261016/validation.json`, `tests_final.log`,
`active_run_integrity.json` and `offline_methods_a/summary.json`.
