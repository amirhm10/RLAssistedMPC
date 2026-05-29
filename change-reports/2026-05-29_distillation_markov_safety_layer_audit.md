# Distillation Markov safety-layer audit

Date: 2026-05-29

## Summary

- Added a Markov-specific safety-layer audit at `report/distillation_markov_safety_layer_audit_2026_05_29.md`.
- The report reviews the May 16, May 18, May 22, May 23, and May 28 Markov evidence.
- It separates layers that should return directly from layers that should return only as softened diagnostics.
- It explicitly documents the nominal copy-paste risk from hard release and hard candidate gates.

## Main conclusion

The next Markov recovery run should restore TD3-priority fallback, reward probation, LS and nominal emergency fallback, executed-action replay, BC handoff, and z trust-region safety. It should not restore the old strict positive prediction-score gate or hard BC release gate as permanent live vetoes, because those layers previously made Markov nearly indistinguishable from nominal MPC.

## Files changed

- `report/distillation_markov_safety_layer_audit_2026_05_29.md`
- `change-reports/2026-05-29_distillation_markov_safety_layer_audit.md`
