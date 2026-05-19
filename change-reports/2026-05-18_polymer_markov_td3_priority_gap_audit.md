# Polymer Markov TD3-Priority Gap Audit

Date: 2026-05-18

## Summary

Audited the latest polymer Markov result at
`Polymer/Results/td3_markov_disturb_zbound_008/20260517_210211/`.

The saved run did not use the intended TD3-priority fallback configuration:

- `td3_priority_fallback = {}`
- `summary_metrics["td3_priority_fallback_enabled"] = False`
- tail-20 TD3 accepted fraction was only `0.1657`
- tail-20 LS fallback fraction was `0.7539`

The old positive prediction-score gate was the main limiter. Tail requested TD3 moves passed drift and cost checks, but only `16.6%` had positive score.

## Changes

- Added notebook config pass-through for `td3_priority_fallback` in:
  - `RL_assisted_MPC_markov_unified.ipynb`
  - `distillation_RL_assisted_MPC_markov_unified.ipynb`
- Added analysis script:
  - `report/scripts/analyze_polymer_markov_td3_priority_gap_20260517.py`
- Added report figures:
  - `report/figures/polymer_markov_td3_priority_gap_20260517/fig_latest_priority_gap_action_sources.png`
  - `report/figures/polymer_markov_td3_priority_gap_20260517/fig_reward_latest_vs_td3_only.png`
- Extended:
  - `report/polymer_markov_td3_only_without_safeguard_2026_05_16.md`

## Validation

- Loaded the saved latest result bundle and diagnostics.
- Verified the saved run had TD3-priority disabled.
- Verified future unified notebooks now pass `CTRL.get("td3_priority_fallback", {})` into `markov_cfg`.
- No notebook execution was performed.
- No Aspen execution was performed.
