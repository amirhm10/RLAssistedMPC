# 2026-05-13 Distillation Markov Latest Analysis

## Scope

- added `report/scripts/generate_distillation_markov_latest_assets.py`
- generated a dated figure bundle under `report/figures/distillation_markov_latest_20260513/`
- added `report/distillation_markov_latest_2026_05_13.md`

## Purpose

This change documents the latest distillation Markov run `20260512_090635` against the disturbance MPC baseline and records the main conclusion clearly:

- the latest run is stable
- it is slightly better than disturbance MPC
- the gain is small enough that the controller remains effectively near-baseline

## Notes

- no controller logic was changed
- the report notes that the saved Markov bundle does not persist a full `config_snapshot` and does not store `markov_z_bound`
