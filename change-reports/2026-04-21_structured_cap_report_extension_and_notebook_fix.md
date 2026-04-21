# 2026-04-21 Structured Cap Report Extension and Notebook Wiring Fix

This update extends the wide-range matrix/structured report and fixes the structured notebook wiring so the intended `A/B` override caps are actually used in future structured runs.

Files updated:

- [report/generate_polymer_wide_range_report.py](../report/generate_polymer_wide_range_report.py)
- [report/polymer_wide_range_matrix_structured_report.md](../report/polymer_wide_range_matrix_structured_report.md)
- [RL_assisted_MPC_structured_matrices_unified.ipynb](../RL_assisted_MPC_structured_matrices_unified.ipynb)
- [distillation_RL_assisted_MPC_structured_matrices_unified.ipynb](../distillation_RL_assisted_MPC_structured_matrices_unified.ipynb)

Report changes:

- embedded figures now render larger in the markdown via explicit HTML image tags
- added a new section comparing the latest polymer matrix rerun (`20260421_174016`) and structured rerun (`20260421_174057`)
- added a new section and figure for the structured asymmetric-cap frontier
- added new report data exports:
  - `wide_range_latest_cap_reruns.csv`
  - `wide_range_structured_asymmetric_cap_frontier.csv`

Main analytical result:

- the latest polymer matrix rerun really used the intended scalar `A` cap and improved further
- the latest polymer structured rerun did **not** use the intended structured `A/B` overrides; its saved structured bounds remained uniform `[0.75, 1.25]`
- so the bad latest structured run is not evidence that the structured `A` cap failed
- Monte Carlo sampling of the structured family shows the intended first structured `A` cap (`1.0566`) is a reasonable admissibility bound when `B` stays wide, while `1.04-1.05` is an even more conservative structured margin

Notebook fix:

- both unified structured notebooks now pull `a_low_override`, `a_high_override`, `b_low_override`, and `b_high_override` from notebook defaults
- both notebooks now pass those values into `build_structured_update_spec(...)`
- both notebooks now also include those values in `structured_cfg`

Validation:

- notebook JSON validated with `nbformat`
- `C:\Users\HAMEDI\miniconda3\envs\rl-env\python.exe -m py_compile report\generate_polymer_wide_range_report.py`
- `C:\Users\HAMEDI\miniconda3\envs\rl-env\python.exe report\generate_polymer_wide_range_report.py`

