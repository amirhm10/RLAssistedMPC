# 2026-04-14 polymer reidentification final surface

Finalized polymer online reidentification as one official workflow and removed the exploratory re-id surfaces.

Changed:
- replaced the old `reid_batch`, `reid_batch_v2`, and `reid_batch_v3_study` defaults families with one `reidentification` family in `systems/polymer/notebook_params.py`
- renamed the active helper and runner modules to `utils/reidentification.py` and `utils/reidentification_runner.py`
- replaced `run_reid_batch_supervisor(...)` with `run_reidentification_supervisor(...)`
- replaced `plot_reid_batch_results(...)` with `plot_reidentification_results(...)`
- removed the ablation-summary and theta-diagnostics public plotting surfaces
- built `RL_assisted_MPC_reidentification_unified.ipynb` from the old v2 launcher and retargeted it to the final fixed-method API
- deleted the old polymer re-id notebooks, study launchers, and root planning markdown
- replaced `report/05_online_reidentification_batch_ridge_blend.tex` with `report/06_polymer_reidentification_final_method.tex`

Final method locked in:
- `block_polymer`
- joint `A+B` identification
- `legacy_previous_measurement` observer alignment
- learned `eta`
- normalized blend extras enabled
- `fro_only` candidate guarding
- `lambda_prev_A=1e-2`, `lambda_prev_B=1e-1`
- `lambda_0_A=1e-4`, `lambda_0_B=1e-3`
- `theta_A in [-0.15, 0.15]`
- `theta_B in [-0.08, 0.08]`
- `delta_A_max = delta_B_max = 0.10`

Validation:
- `py_compile` on the renamed source files and polymer notebook defaults
- defaults lookup checks for the new key and the removed keys
- short `rl-env` polymer smoke run through `run_reidentification_supervisor(...)`
- plotting smoke with `plot_reidentification_results(...)` using both default plots and `debug_id_plots=True`
- notebook JSON parse for `RL_assisted_MPC_reidentification_unified.ipynb`
