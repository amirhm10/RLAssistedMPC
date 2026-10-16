# Recovery after the Codex desktop restart

The author reported that restarting Codex closed the newly launched studies and
Aspen instances. The process inventory confirmed that only the manually started
distillation repetition parent, its ten workers and ten Aspen instances survived.
Nine other study workers, their supervisors/consoles and cases 11--16 were gone.
Their on-disk results were not deleted.

The previous launcher used `CREATE_NEW_CONSOLE`, which provides a visible window
but retains inherited Windows job membership. An escalated diagnostic process
reported job limits `0x2800`, including kill-on-job-close. A breakaway probe still
belonged to a Windows job. The simultaneous loss of app-launched processes and
survival of the manually launched repetition tree is consistent with app-owned
process cleanup, rather than nine independent controller failures. The exact
termination event was not recorded before the old app exited.

`review_restart_recovery.py` now delegates this immediate recovery to the Windows
Task Scheduler service using the existing user's interactive token/SID. It stores
no password, requests no elevated run level, defines no recurring trigger, and
sets the execution time limit to `PT0S`. The independent broker remains alive
while the nine visible CMD sessions remain open. A task-owned Windows job is
expected; the isolation contract verifies the pre-existing Schedule service is
the broker's parent and that it runs in the user's nonzero desktop session.

The diagnostic task passed under service PID 2524, session 2. Production task
`RLAssistedMPC_ReviewerRecovery_1792172343082316600` launched broker PID 43188.
All nine CMD sessions opened at **13:39:11 Toronto, October 16**. They use the
original `rl-assisted-env` interpreter and existing workers. The case-11 E3
wrapper only adds live worker-log streaming and the existing numerical-runtime
mitigation; it retains all saved scientific settings and worker code.

| Recovered study | Recovery behavior | New worker PID |
|---|---|---:|
| Polymer E1 combined | Full-state checkpoint after cycle 10; resumes cycle 11 | 16680 |
| Polymer E2 combined SG | Restart original 200-cycle protocol | 46320 |
| Polymer repaired ARX | Restart original 200-cycle protocol | 36936 |
| Column E3 combined, Aspen 11 | New a2, original 300-cycle protocol; reuse completed nominal/Markov | 49828 |
| Column combined SG, Aspen 12 | New a2, original 200-cycle protocol | 27148 |
| Column threshold, Aspen 13 | New a2, original calibration/main protocol | 16284 |
| Column LS-only, Aspen 14 | New a2, original 200-cycle protocol | 37868 |
| Column two-weight BO, Aspen 15 | New a2, original 40-trial calibration/main protocol | 51024 |
| Column E1 restoration pilot, Aspen 16 | New a2, original 400-step prefix and three 24-step continuations | 1752 |

E1's 823,810,582-byte checkpoint checksum and source inventory verified before
restart. Its log confirms restoration after step 8,000 and subsequent cycle 11.
The existing manager archives the post-checkpoint decision suffix before replay.
The interrupted run had reached cycle 13 and 30 matched decisions. Polymer SG
had reached cycle 36, repaired ARX cycle 130 and column E3 combined cycle 33.
Those three have no verified full-state training/simulator resume, so their
incomplete trajectories are not spliced into restarted runs. The five newly
started column jobs had not completed a main training cycle before interruption.

Original checkpoint slots and scientific attempts remain in place. Top-level
polymer metadata/logs and column/E3 manifests were copied into the dedicated
`preserved_before_restart` directory before any worker restarted. Column/E3
continuations use new a2 directories; completed nominal/Markov artifacts verify
and are reused. Old attempt files may retain historical `running` status because
their parents were terminated externally; the new attempt and recovery records
determine current state.

Seven focused regression checks pass, covering scheduler-owned interactive
context, rejection of app/session-zero contexts and destinations outside the
recovery workspace, and preservation of logs/checkpoints/raw attempt data.
The live probe and actual service-parent process tree validate the launch path.
No deliberate second Codex restart was used as a destructive training test.
The broker's lifetime is independent of Codex; ordinary Windows shutdown/logoff,
closing a job's CMD or stopping its task can still interrupt execution.

Active original source inventories and all 46 frozen manuscript files verify
unchanged. No original controller, learner, simulator adapter, reward, profile,
TeX or PDF was edited. The unrelated user edit to
`run_polymer_phase2_repeats.py` is excluded. The surviving ten repetitions were
neither restarted nor interrupted.

Recovery directory: `Results/restart_recovery/20261016_codex_restart`.
Bounded audit/evidence: `change-reports/codex_restart_recovery_20261016/`.
The original scientific source guards remain authoritative; launching is not a
completed-performance claim or reviewer-comment closure.

Inspected: live process trees/Windows job flags, interactive scheduler metadata,
original launch commands and recovery code, checkpoint checksum/source inventory,
attempt/status/log files, completed E3 inventories and the manuscript freeze.
Changed: the isolated launcher, its regression tests, this report and bounded
evidence; normal new outputs are generated only by the authorized study workers.

Official Windows sandbox documentation also warns about child-process lifetime
limits in some execution modes; it does not establish the exact prior app-shutdown
event. The diagnosis above primarily rests on this machine's process evidence:
[OpenAI Windows documentation](https://learn.chatgpt.com/docs/windows/windows-sandbox).
