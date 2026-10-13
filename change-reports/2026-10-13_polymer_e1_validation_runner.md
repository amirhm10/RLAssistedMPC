# Polymer reviewer E1 runner — 2026-10-13

Amir authorized code development with one seed and explicitly required reuse of the existing training procedure, including warm-up and hand-off. Added a separate E1 runner/package, focused tests and usage/protocol documentation. No pre-existing scientific source, default, data asset or running repetition-wrapper file was changed by this task.

Default production plan: five gated configuration runs at seed 7; 100 matched decisions per channel per operating window; four standalone channels plus four combined contexts; 3,200 matched decisions. Main training preserves the original Scenario 2 schedule. Branches alone freeze networks/replay/exploration and normalizer statistics.

The new process-local AST adapter suspends the existing runner at gate assignments and reuses its original continuation statements. It handles the Markov pending-replay update before the gate, horizon action holds, existing guards and coupled downstream recalculation. Captured random state and network modes are restored before main training continues.

Also added common-input, held-out-output linear-predictor diagnostics at dynamic-matrix decisions, physical per-output metrics, decision-time replay support, local margin re-selection, CPU inference snapshots and isolated snapshot replay. These supply evidence for parts of E3–E5 without claiming full joint-decision analysis, closed-loop stress testing, seed robustness, or formal guarantees.

The integration tests caught and corrected an E1 defaults-adapter bug: the combined entrypoint has no NB_CONFIGURE hook. E1 now uses the same process-local defaults-provider interface as the existing repeat wrapper. The original combined source was not edited.

Verification is recorded in the final development handoff. Full E1 production training was not launched. Compressed-clock technical checks are explicitly labelled and excluded from scientific evidence.

All five SG training-preservation cases passed. CPU inference snapshots reproduced the original paired outputs, inputs, rewards and references for weights, Markov and combined. The original-network CUDA CLI saved 4 horizon and 16 combined decisions with zero failed pairs. The independent prediction oracle passed. SHA-256 audits preserved all 124 pre-existing tracked code/data/workflow files and all 46 original manuscript files. No TeX source was edited.

The final 25-case suite reported 24 passes and one Markov saved-replay failure. Diagnosis found that Python's globalized helper definitions can have bare qualified names; restoration therefore missed their private namespace and retained the captured clock. Private helpers now carry their runner ownership and are also recognized by their source range. The targeted post-fix run passed 19 tests, including a new serialized-clock regression and the affected Markov end-to-end case. All 24 saved test snapshots (48 branches) were then replayed and matched the original inputs, outputs, rewards and references. Unaffected training-parity cases were not needlessly rerun.
