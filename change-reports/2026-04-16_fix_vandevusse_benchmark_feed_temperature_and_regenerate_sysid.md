## Summary

Fixed the Van de Vusse benchmark inconsistency by restoring the control-benchmark feed temperature and then regenerating the canonical Van de Vusse system-identification artifacts before rerunning the baseline MPC path.

## Root Cause

- The Van de Vusse nonlinear plant equations were already broadly consistent with the benchmark conventions.
- The main inconsistency was that the default feed temperature had been taken from a later modeling/PINN source (`378.1 K`) instead of the control benchmark (`130.0 C = 403.15 K`).
- That wrong default made the benchmark operating point non-stationary for the implemented nonlinear plant and shifted the solved steady state far away from the intended benchmark temperature.

## What Changed

- Updated `systems/vandevusse/config.py` so the control-benchmark literature is the authoritative default source.
- Restored the benchmark feed temperature to `403.15 K`.
- Kept the later `378.1 K` design point only as a secondary legacy cross-check.
- Added reusable benchmark residual / consistency diagnostics in the Van de Vusse plant and system-ID helper layers.
- Updated the Van de Vusse system-identification notebook to show legacy-vs-active benchmark consistency before artifact export.
- Updated the Van de Vusse reports so the benchmark fix and regenerated-ID dependency are documented.

## Regenerated Artifacts

After the benchmark-default fix, the canonical Van de Vusse workflow regenerated:

- `VanDeVusse/Data/system_dict.pickle`
- `VanDeVusse/Data/scaling_factor.pickle`
- `VanDeVusse/Data/min_max_states.pickle`
- `VanDeVusse/Data/system_id_metadata.pickle`
- `VanDeVusse/Data/system_id_metadata.json`

The Van de Vusse baseline MPC path was rerun only after those refreshed system-ID artifacts were written.

## Scope

- Polymer was not changed.
- Distillation was not changed.
- The direct-local-linearization system-identification method was kept; only the benchmark plant/default consistency was corrected before regeneration.
