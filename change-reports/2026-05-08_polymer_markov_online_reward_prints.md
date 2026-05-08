# Polymer Markov Online Reward Prints

## Summary

Aligned the polymer Markov runner with the unified runner convention for online progress output.

## Notes

- The live Markov loop now prints `Sub_Episode: ... | avg. reward: ... |` at each completed subepisode boundary.
- The print includes Markov-specific diagnostics: accepted fraction, TD3 accepted fraction, LS fallback fraction, nominal fallback fraction, and average executed `z`.
- The runner still returns and saves the episode-average reward table for post-run inspection.

## Validation

- Syntax validation passed for `report/scripts/generate_polymer_markov_correction_assets.py`.
- A small override run with short setpoint blocks printed subepisode average rewards during the online loop.

