# 2026-04-21 Wide-Range Report Extension: Warm Start, Observer Refresh, and Exploration Noise

This update extends [report/polymer_wide_range_matrix_structured_report.md](../report/polymer_wide_range_matrix_structured_report.md) and its generator [report/generate_polymer_wide_range_report.py](../report/generate_polymer_wide_range_report.py) with three method-level sections requested after the wide-range matrix/structured analysis:

- shifted MPC warm-start analysis across all method families, not only matrix/structured
- periodic observer redesign / observer-refresh analysis
- exploration-noise analysis tied to the current notebook defaults and saved runs

What was added:

- new report sections:
  - `Will Shifted MPC Warm Start Help Across Methods?`
  - `Would Periodic Observer Updating Help?`
  - `Is The Current Exploration Noise Making The Wide-Range Problem Worse?`
- new figures:
  - `wide_range_warmstart_exploration.png`
  - `wide_range_observer_refresh_support.png`
- new data exports:
  - `wide_range_method_defaults.csv`
  - `wide_range_method_diagnostics.csv`
  - `wide_range_observer_refresh.csv`

Main analysis points now documented in the report:

- shifted warm start is already implemented across the MPC-solving runners, but left off by default
- warm start looks most promising for residual, matrix, structured, weights, and reidentification, and least promising for horizon because the MPC dimension changes frequently
- periodic observer redesign is not a blanket fix; it is more defensible when sustained model drift exists and the refreshed model is trustworthy
- polymer wide matrix/structured runs show materially larger model drift than the current saved distillation matrix/structured runs
- current polymer reidentification is not a good observer-refresh candidate because candidate-valid and update-success fractions are extremely low
- distillation reidentification is materially healthier than polymer on candidate validity and conditioning, so it is a better first place to study guarded observer refresh
- the current continuous families all use the same param-noise schedule despite much wider model authority, which likely accentuates the early deterioration seen in the wide runs

Validation:

- `C:\Users\HAMEDI\miniconda3\envs\rl-env\python.exe -m py_compile report\generate_polymer_wide_range_report.py`
- `C:\Users\HAMEDI\miniconda3\envs\rl-env\python.exe report\generate_polymer_wide_range_report.py`

