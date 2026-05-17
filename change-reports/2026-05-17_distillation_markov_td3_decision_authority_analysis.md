# 2026-05-17 Distillation Markov TD3 decision-authority analysis

## What changed

- added `report/scripts/generate_distillation_markov_td3_decision_authority_assets_20260517.py`
- generated a new analysis asset set under `report/figures/distillation_markov_td3_decision_authority_20260517/`
- added `report/distillation_markov_td3_decision_authority_2026_05_17.md`

## Why

The earlier family report showed that the TD3-only distillation Markov run was the only branch where TD3 was truly active online.

This follow-up note goes deeper into the decision logic itself:

- why guarded and relaxed variants still fall back almost all the time
- which gates are actually suppressing TD3
- why the old safeguards are misaligned with the best TD3-only run
- what redesign would make TD3 the main live decision-maker while keeping LS and nominal MPC as controlled fallback tools

## Scope

- analysis only
- no controller code changed
- no raw result bundles were overwritten
- conclusions are based on the latest completed saved Markov runs from 2026-05-16, not the currently running May 17 TD3-only retune

