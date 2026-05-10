# Distillation Reward Geometry Smoothing Report Extension

Date: 2026-05-10

## Summary

- added a reward-geometry smoothing analysis for the distillation matrix-family report
- generated a new figure bundle under `report/figures/distillation_reward_geometry_smoothing_20260510/`
- extended the report with concrete smoothing levers, quantitative candidate ratios, and fixed-trajectory rescoring on the May 8 observer-refresh runs

## Main Finding

Lowering distillation `Q1` is the cleanest first lever because it reduces both the edge-slope and bonus ratios directly. Lowering `beta` and replacing the exponential bonus with a smoother shape help with absolute harshness and reward brittleness, but they do not fix the cross-output ratio by themselves. Fixed-trajectory rescoring narrows the May 8 reward deficits substantially, but does not turn those saved trajectories into wins against disturbance MPC.

## Files Changed

- `report/distillation_matrix_structured_step4g_latest_2026_05_04.md`
- `report/scripts/generate_distillation_reward_smoothing_assets.py`
- `report/figures/distillation_reward_geometry_smoothing_20260510/`
