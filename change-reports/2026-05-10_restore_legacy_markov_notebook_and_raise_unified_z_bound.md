## Summary

Added a separate legacy polymer Markov notebook and matching legacy runtime script, both restored from the commit immediately before the unified Markov migration (`ce34e86^`), so the old prototype path can be run side-by-side with the current unified notebook.

## Details

- Added `polymer_markov_corrected_mpc_legacy.ipynb`.
- Added `report/scripts/generate_polymer_markov_correction_assets_legacy.py`.
- Kept the legacy path on its historical `z_bound = 0.05`.
- Increased the current unified Markov default `z_bound` from `0.05` to `0.10` in `systems/polymer/notebook_params.py`.

## Verification

- Loaded both notebooks as valid JSON.
- Imported `build_config()` from the restored legacy script successfully.
- Confirmed:
  - legacy notebook/script default `z_bound = 0.05`
  - current unified Markov default `z_bound = 0.10`
