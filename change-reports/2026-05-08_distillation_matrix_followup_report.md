# Distillation Matrix Follow-Up Report

Date: 2026-05-08

## Summary

- added `report/scripts/generate_distillation_matrix_structured_followup_assets.py` to reproduce the May 8 distillation matrix-family follow-up analysis
- generated new figures and CSV summaries under `report/figures/distillation_matrix_structured_followup_20260508/`
- rewrote `report/distillation_matrix_structured_step4g_latest_2026_05_04.md` as a true follow-up note that compares May 3 vs May 8 distillation runs, contrasts them with the latest polymer matrix-family references, and evaluates the Markov-correction direction

## Key Findings Captured

- `decision_interval = 20` did not rescue the distillation scalar or structured matrix runs
- the latest distillation runs remain decisively worse than disturbance MPC on reward, output-2 MAE, and input movement
- polymer scalar matrix remains a valid positive counterexample, so the failure is system-specific rather than a universal matrix-family failure
- the report now argues that a low-dimensional, prediction-error-gated distillation Markov pilot is a better next experiment than another wide matrix rerun
