## Summary

Extended `report/polymer_markov_latest_run_analysis_2026_05_09.md` with a dual-notebook follow-up comparing:

- the newest unified Markov notebook run
- the newest restored legacy Markov notebook run
- the original older legacy prototype run

## Main findings

- The restored legacy notebook is already close to the original prototype behavior.
- The unified notebook still executes a meaningfully different policy.
- The biggest remaining differences are the unified nominal-reference solve, larger `z` authority, much higher nominal fallback rate, and the use of the canonical saved baseline as the report comparator.

## Artifacts

- Added `report/scripts/analyze_polymer_markov_dual_notebook_followup.py`
- Added figures and CSV summaries under `report/figures/polymer_markov_dual_notebook_followup_20260510/`
