# 2026-05-09 Markov prototype nominal-solver follow-up analysis

## Objective

Analyze the newest polymer Markov rerun after restoring the prototype-style nominal online solve default and extend the running Markov report with figures and quantitative comparisons.

## Files changed

- `report/scripts/analyze_polymer_markov_prototype_nominal_solver_followup.py`
- `report/polymer_markov_latest_run_analysis_2026_05_09.md`
- `report/figures/polymer_markov_prototype_nominal_solver_followup_20260509/*`

## What the analysis added

- Quantified the newest run `Polymer/Results/td3_markov_disturb/20260509_184140/input_data.pkl`.
- Compared it against:
  - the previous unified prototype-reward rerun `20260509_155540`
  - the old prototype run `20260508_123902`
  - the canonical baseline `Polymer/Data/mpc_results_dist.pickle`
- Generated new reward-window, reward-component, action-mix, solver-switch, and final-episode difference figures.
- Extended the main Markov report with a dedicated follow-up section for the lifted-`G0` nominal-solver rerun.

## Main finding

Restoring the prototype nominal solve did not restore the old prototype behavior. The newest run remains visually close to canonical MPC, and under the prototype reward it is actually much worse than the canonical baseline because the old exponential inside-band bonus is extremely sensitive to small trajectory shifts.
