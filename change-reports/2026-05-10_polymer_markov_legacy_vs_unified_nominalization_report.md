# 2026-05-10 Polymer Markov Legacy-vs-Unified Nominalization Report

Added a new standalone comparison report for the latest legacy and unified polymer Markov runs:

- `report/polymer_markov_legacy_vs_unified_where_unified_becomes_nominal_2026_05_10.md`

Added a reproducible analysis script and regenerated report figures:

- `report/scripts/analyze_polymer_markov_latest_legacy_vs_unified.py`
- `report/figures/polymer_markov_legacy_vs_unified_20260510/`

Main conclusion:

- the current gap is not primarily reward mismatch anymore
- the largest behavioral difference is still the execution protocol, especially `warm_start = 10` in the legacy path versus `warm_start = 0` in the unified path
- the legacy run that looks better is still heavily LS-teacher-driven rather than clearly TD3-driven
