# 2026-05-10 Polymer Markov Legacy Mimic Ladder Surface

Added a new isolated polymer Markov experiment surface for stepwise unified-to-legacy mimic work.

Created:

- `utils/markov_runner_legacy_mimic.py`
- `polymer_markov_corrected_mpc_unified_legacy_mimic.ipynb`
- `report/scripts/analyze_polymer_markov_legacy_mimic_ladder.py`
- `report/polymer_markov_legacy_mimic_ladder_2026_05_10.md`

Main behavior:

- the new runner starts from the current unified Markov logic
- ladder steps are selected by `legacy_mimic_step`
- Steps 0 through 5 are implemented as cumulative deltas
- saved result bundles now preserve legacy-mimic metadata for reporting and analysis

This is intentionally Markov-only. Shared-runner generalization for other notebook families is deferred until the mimic ladder identifies proven causal differences.
