# Reduce Distillation Reward Q1 To 5300

Date: 2026-05-10

## Summary

- reduced the shared distillation relative-band reward output-1 weight from `10000` to `5300`
- left `beta = 3` unchanged from the previous reward-smoothing update
- kept the band definitions, gate, and bonus type unchanged so only the output-1 weight moved

## Files Changed

- `systems/distillation/config.py`
