## Summary

Restructured the `StatsControl2026` Project 2 section into two introduction slides followed by one slide per completed single-agent polymer supervisor.

## Updated

- `StatsControl2026/stats_control_2026_slides.tex`
- `utils/polymer_multiseed_core_study.py`
- `report/figures/polymer_five_seed_core_study_horizon_dueling/20260503_151836/fig_reward_mean_std_by_method.png`
- `report/figures/polymer_five_seed_core_study_matrix/20260503_170045/fig_reward_mean_std_by_method.png`
- `report/figures/polymer_five_seed_core_study_weights/20260503_173957/fig_reward_mean_std_by_method.png`
- `report/figures/polymer_five_seed_core_study_residual/20260503_181944/fig_reward_mean_std_by_method.png`

## Slide additions

- Kept the Project 2 architecture slide that defines the two case studies, restates the nominal OF-MPC objective, and positions RL as a supervisory adaptation layer around MPC.
- Reworked the second Project 2 slide into a compact overview of the four supervisory families plus the latest exported multiseed polymer ranking.
- Added four method-specific slides:
  - horizon agent
  - matrix agent
  - weight agent
  - residual agent
- Each method slide now uses:
  - left column: method explanation with math and interpretation
  - right column: latest reward-learning curve and last-episode MPC comparison figure
- Used the latest completed polymer multiseed exports from May 3, 2026 for horizon, matrix, weights, and residual.
- Replaced the old combined placeholder with a real combined-agent slide using the latest completed polymer multiseed export from `report/figures/polymer_five_seed_core_study_combined/20260503_185939`.
- Added a final disturbance-mode summary slide that compares all five agent families against the same disturbance MPC baseline and includes per-method episode win counts against MPC.
- Updated the multiseed reward summary figure so the slide-facing reward plots:
  - exclude episode 1 on both RL and MPC traces
  - overlay the saved baseline MPC reward from `Polymer/Data/mpc_results_dist.pickle`
  - preserve the existing method mean plus/minus standard deviation shading
- Revised Slides 16 and 17 after code and report review so that:
  - Slide 16 is now a general Project 2 overview in bullet form rather than a repeat of control-setting details
  - Slide 16 explicitly notes that the replay workflow and reward function are reused from Project 1
  - Slide 17 now focuses on the actual method formulation: shared state, mismatch augmentation, shared reward/replay tuple, supervisory action mappings, and the common rollout algorithm
  - Slide 17 cites the active implementation surfaces used to reconstruct the method (`state_features.py`, `rewards.py`, `agent_step_runtime.py`, and the single-agent runners)
- Added two closing Project 2 slides:
  - a combined-agent slide with the latest reward and last-episode disturbance-mode results
  - a disturbance-baseline summary table across horizon, matrix, weights, residual, and combined
- Verified from the saved compare bundles that every method on the new summary slide uses `compare_mode = disturb`.
- Counted `win episodes` as the number of episodes `2:200` for which the RL-assisted average episode reward exceeded the disturbance MPC reward, aggregated across the three saved seeds.
- Added a final-reward percentage-improvement column relative to the same disturbance MPC reward baseline and surfaced the combined agent's `+52.5%` final reward improvement directly on its method slide.
- Localized every figure used by the deck into `StatsControl2026/figures/` and updated the slide source to use only local figure paths.
- Cleaned the `StatsControl2026/` root so it now contains only the slide source, rendered PDF, and the organized `figures/` directory.

## Validation

- Regenerated the four slide-facing polymer reward summary figures from their saved multiseed manifests and the existing baseline MPC pickle.
- Rebuilt `StatsControl2026/stats_control_2026_slides.pdf` locally with MiKTeX `pdflatex`; the deck compiled successfully to 23 pages.
- The rebuild still reports several overfull box warnings on the dense Project 2 frames, but there are no LaTeX errors preventing PDF generation.
- Removed the generated LaTeX scratch files from `StatsControl2026/` after the successful build to keep the folder tidy.
