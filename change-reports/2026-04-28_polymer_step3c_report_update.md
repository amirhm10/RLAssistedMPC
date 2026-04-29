# 2026-04-28 Polymer Step 3C Report Update

## Summary

Updated the main matrix report with the latest polymer Step 3C shadow-study scalar and structured runs.

## Added analysis

- Confirmed the study configuration from the saved bundles:
  - Step 2 on
  - Step 4 off
  - Step 3B hard fallback off
  - Step 3C shadow on
- Compared the Step 3C study runs against the current Step 4G defaults.
- Added reward-window tables and figures.
- Added shadow diagnostic tables and figures for:
  - nominal penalty
  - safe threshold
  - candidate advantage
  - safe / benefit / dual pass rates
- Added episode-level correlation analysis showing that the current safe-pass test is negatively aligned with later positive reward episodes.

## Outcome

The report now lands on:

- keep Step 4G as the polymer default;
- keep Step 3C shadow-only for now;
- use Step 3C shadow logging, not a hard gate, for the next distillation study.
