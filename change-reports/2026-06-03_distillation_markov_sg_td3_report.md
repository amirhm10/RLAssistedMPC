# Distillation Markov SG-TD3 Report

Date: 2026-06-03

## Summary

Added a standalone analysis report for the latest distillation Markov SG-TD3 critic-warm-3 LS-or-MPC run from `20260602_192543`.

## Changes

- Added `report/scripts/analyze_distillation_markov_sg_td3_20260603.py`.
- Added `report/distillation_markov_sg_td3_outcome_2026_06_03.md`.
- Generated local ignored figures and CSV summaries under `report/figures/distillation_markov_sg_td3_20260603/`.

## Findings

- SG-TD3 tail-20 reward is `26.62` versus OF-MPC `6.39`.
- SG-TD3 final reward is `28.43` versus OF-MPC `6.93`.
- SG-TD3 avoids the TD3-full no-safeguard early release crash. Its worst first-20 post-warm reward is `3.67`, while TD3-full no-safeguard reaches `-113.80`.
- SG-TD3 tail T85 MAE is `0.0669` versus OF-MPC `0.1921`.
- The SG gate selects TD3 on `38.1%` of tail steps and keeps supervisor action on `61.9%`.

## Validation

- Ran the analysis script with `C:\Users\hamediaa\.conda\envs\rl-env\python.exe`.
- Generated the requested figure pack, CSV summaries, and manifest.
- No Aspen simulation was launched.
