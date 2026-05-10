# Reduce Distillation Reward Q1 And Beta

Date: 2026-05-10

## Summary

- reduced the shared distillation relative-band reward output-1 weight from `37000` to `10000`
- reduced the shared distillation relative-band reward bonus scale `beta` from `7` to `3`
- left the band definitions, gate, and bonus type unchanged so this is a clean two-knob reward smoothing change

## Files Changed

- `systems/distillation/config.py`
