# Clarify Rho And Distillation Sensitivity

Date: 2026-04-24

## Summary

- Added a plain-language explanation of `rho(A)` as spectral radius in the matrix-cap report.
- Added the eigenvalue math used to calculate `rho(A)` and the scalar `A` multiplier cap.
- Clarified that polymer success includes the wide `A,B` matrix search, while distillation remains sensitive even after `A` is tightened because `B` changes the finite-horizon input-output map used by MPC.
