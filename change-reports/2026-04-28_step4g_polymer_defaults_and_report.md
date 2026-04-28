# 2026-04-28 Step 4G Polymer Defaults And Report Update

## Summary

Implemented Step 4G as the new polymer default handoff for scalar and structured matrix supervisors, while leaving distillation disabled by default. Also extended the matrix multiplier report with the Step 4G phase definition, equations, and default-policy rationale.

## What Changed

- Polymer scalar matrix defaults now combine:
  - stronger Step 4 BC
  - a short hidden freeze
  - light Step 2 release-protected advisory caps
  - Step 3 fallback still off
- Polymer structured matrix defaults now combine:
  - weighted Step 4E BC
  - a short hidden freeze
  - a slightly stricter light Step 2 release-protected advisory-cap schedule
  - Step 3 fallback still off
- Distillation matrix and structured-matrix defaults remain unchanged, with BC still disabled by default.
- The polymer unified matrix notebooks now show a derived handoff label and include BC, release-guard, and fallback state in the run summary.
- The main matrix multiplier report now documents Step 4G as an implemented default policy in polymer, with math and explanation only; no Step 4G result claims were added.

## Validation

- Loaded polymer and distillation notebook defaults to confirm the new enablement matrix.
- Validated the two edited polymer notebooks with `nbformat`.
- Confirmed the polymer notebook summary cells expose the derived Step 4G handoff label before execution.
