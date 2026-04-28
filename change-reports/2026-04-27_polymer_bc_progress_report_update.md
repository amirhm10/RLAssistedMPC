# 2026-04-27 Polymer BC Progress Report Update

Updated `report/matrix_multiplier_cap_calculation_and_distillation_recovery.md` with the latest polymer matrix and structured-matrix BC-only run analysis.

Included:

- explicit verification that the saved 2026-04-27 polymer runs are BC-only, with Step 2/3 and post-warm-start freeze disabled;
- reward-window comparison against the 2026-04-24 hidden-release runs and the 2026-04-25 Step 3B stack;
- BC-only handoff diagnostics and multiplier-window summaries;
- new supporting figures and CSV summaries under `report/figures/matrix_multiplier_bc_progress_20260427/`.

Main conclusion:

- scalar BC-only preserves the late matrix benefit and slightly improves the full run over the older scalar hidden-release baseline, but it still makes the first live window worse;
- structured BC-only remains too weak as a handoff replacement because the early window degrades too much relative to the older hidden-release baseline;
- the shared BC surface is useful, but polymer still needs a stronger handoff design before BC should be transferred to distillation defaults.
