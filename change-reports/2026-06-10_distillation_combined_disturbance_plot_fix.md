# Distillation Combined Disturbance Plot Fix

Date: 2026-06-10

## Summary

Fixed the distillation combined runner failure that occurred after a completed long Aspen rollout during result plotting:

```text
AttributeError: 'str' object has no attribute 'keys'
```

## Cause

`run_combined_supervisor()` already stores `disturbance_profile` as a plotting dictionary derived from the disturbance schedule. The root distillation combined runner then overwrote that dictionary with the string profile name, such as `"fluctuation"`, immediately before calling `plot_combined_results()`.

`utils.plotting_core.disturbance_plot_items()` expected a dictionary and tried to call `.keys()` on the string.

## Fix

- Keep the schedule-derived `result_bundle["disturbance_profile"]` intact.
- Store the string profile name separately as `result_bundle["disturbance_profile_name"]`.
- Make `disturbance_plot_items()` defensive:
  - string profile names are skipped instead of crashing,
  - 1D and 2D numeric schedules are converted into plot items,
  - dictionary profiles continue to work as before.
- Save the combined `input_data.pkl` immediately after creating the output directory, before any figure generation. The final save still runs after plotting when plotting succeeds, but a late plotting failure no longer loses the completed rollout bundle.

## Verification

- Python syntax checks for the modified runner, plotting helper, and tests.
- `tests/test_plotting_disturbance_items.py`
- `tests/test_distillation_combined_runner.py`
- Added a regression test that forces combined figure generation to fail and verifies `input_data.pkl` already exists.
