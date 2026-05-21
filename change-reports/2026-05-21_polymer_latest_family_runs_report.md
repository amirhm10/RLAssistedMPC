# Polymer Latest Family Runs Report

Date: 2026-05-21

## Summary

Generated an extensive analysis report for the latest disturbed polymer runs covering OF-MPC, horizon DQN, dueling horizon DQN, TD3 weights, TD3 residual, TD3 Markov, and the combined horizon + Markov + weights + residual supervisor.

## Outputs

- Report: `report/polymer_latest_family_runs_2026_05_21.md`
- Analysis script: `report/scripts/analyze_polymer_latest_family_runs_20260521.py`
- Figures and tables: `report/figures/polymer_latest_family_runs_20260521/`

## Main Finding

The combined supervisor is the strongest latest run by tail reward and tail tracking error. Markov-only remains safe under the polymer z-safety layer, but its high projection-active fraction suggests the raw TD3 Markov request is often being clipped or projected before execution.

## Validation

The figures were visually checked for readability. The report generator was rerun after wording cleanup so the markdown and saved figures are reproducible from the script.
