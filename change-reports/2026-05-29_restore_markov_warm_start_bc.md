# Restore Markov warm-start BC while keeping post-warm handoff

Date: 2026-05-29

## Summary

- Restored Markov behavioral-cloning loss during the 10 warm-start subepisodes only.
- Kept the post-warm raw-action handoff as the release smoother.
- Added an explicit `handoff.start_after_warm_start` option so BC training and handoff can use different windows.

## Rationale

The prior five-layer restoration correctly avoided post-warm LS copy-paste, but it also removed useful warm-start BC pretraining. This update restores the old warm-start teacher while preventing the teacher from continuing through the live TD3 release.

## Validation

- `py_compile` passed for Markov config/helper/runner files.
- No-Aspen Markov safety config smoke check passed.
- Markov BC schedule smoke check confirmed BC active on steps `0..3999` and handoff starting at step `4000`.
