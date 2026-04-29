# 2026-04-28 Step 3D Usefulness Gate For Distillation Structured Matrix

## Summary

- added `mpc_usefulness_gate` defaults to the polymer and distillation matrix-family config surfaces;
- implemented shared Step 3D hard gating in `utils/mpc_acceptance_gate.py`;
- wired Step 3D through the scalar and structured matrix runners with dedicated logs and mutual-exclusion validation against Step 3B and Step 3C;
- updated the four unified matrix notebooks to load and display the new Step 3D block;
- removed the polymer-local Step 3C study override so polymer returns to shared Step 4G defaults;
- enabled Step 3D by default only for `distillation structured_matrix`.

## Notes

- Step 3D evaluates the Step 2-clipped candidate, not the raw policy request.
- Step 2 is required when Step 3D is enabled.
- Polymer remains on Step 4G by default.
