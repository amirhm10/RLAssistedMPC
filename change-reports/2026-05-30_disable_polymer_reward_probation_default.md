# Disable Remaining Polymer Reward Probation Default

Date: 2026-05-30

## Summary

Disabled the remaining reward-probation default in the polymer Markov priority fallback helper so no notebook default in the polymer or distillation families enables reward-collapse probation.

## Validation

- Compiled `systems/distillation/notebook_params.py` and `systems/polymer/notebook_params.py`.
- Walked all distillation and polymer notebook defaults and asserted every `reward_probation.enabled` value is `False`.
- Kept reward-probation code and log schemas intact for compatibility, but inactive by default.
