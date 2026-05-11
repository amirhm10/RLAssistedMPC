# Polymer Markov Legacy Mimic Ladder Results

Date: 2026-05-11

## What changed

- updated `report/scripts/analyze_polymer_markov_legacy_mimic_ladder.py` so it can analyze all mimic steps in one pass and load the saved result bundles robustly
- regenerated the per-step and all-step ladder summaries under `report/figures/polymer_markov_legacy_mimic_ladder_20260510/`
- extended `report/polymer_markov_legacy_mimic_ladder_2026_05_10.md` with the latest Step 0 through Step 6 results, causal verdicts, transfer guidance, and the recommended Step 7 release-schedule experiment

## Main finding

- Step 3 plant-step / disturbance parity is the only large causal controller change
- Steps 1 and 2 are effectively already aligned in the current unified Markov path
- Step 5 is useful for report parity, not control-law parity
- the next remaining mechanism to test is the legacy-style LS-only early release schedule

## Files changed

- `report/scripts/analyze_polymer_markov_legacy_mimic_ladder.py`
- `report/polymer_markov_legacy_mimic_ladder_2026_05_10.md`
