## Summary

Fixed the residual plotting failure in the final cell of the unified residual notebook.

## Root Cause

`utils/plotting_core.py` used invalid mathtext legend labels:

- `r"$\\rho$"`
- `r"$\\rho_{eff}$"`

Matplotlib expects a single backslash inside the raw string for math mode commands, so saving the residual rho figure raised a `ValueError` while rendering the legend.

## Change

Updated the residual rho trace labels to valid mathtext:

- `r"$\rho$"`
- `r"$\rho_{\mathrm{eff}}$"`

## Validation

- `python -m py_compile utils/plotting_core.py`
- direct matplotlib render check under `rl-env` using the corrected labels
