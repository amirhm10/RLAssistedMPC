# Restore Distillation Default Observer Poles To `p00`

Date: 2026-04-29

## Summary

Reverted the shared distillation observer-pole default from the later `p19`-style uniform fast set back to the earlier `p00_old_aggressive_reference` set.

## Files Changed

- `systems/distillation/config.py`

## Notes

- The restored default pole vector is:
  `[0.0115, 0.0320, 0.0350, 0.0410, 0.0419, 0.0748, 0.4104]`
- Distillation notebook families read these poles through the shared `DISTILLATION_OBSERVER_POLES` constant, so this change affects the default baseline and RL notebook surfaces without separate notebook edits.
