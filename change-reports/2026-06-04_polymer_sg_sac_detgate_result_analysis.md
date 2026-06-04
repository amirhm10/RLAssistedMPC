# Polymer SG-SAC Detgate Result Analysis

## Summary

- Extended the polymer SG-SAC/SG-DQN finished-run report with a follow-up on the new SG-SAC `detgate_hidden7` runs.
- Added a reproducible analysis script comparing residual, weights, and Markov `detgate_hidden7` SG-SAC against their pre-detgate `critic_warm3` SG-SAC counterparts.
- Generated local figures and CSV/JSON artifacts under `report/figures/2026-06-04_polymer_sg_sac_detgate_hidden7/`.

## Main Findings

- Residual SG-SAC shows the clearest benefit: first-live-10 reward improves from `-10.49` to `-4.77`, and tail reward/tracking improve slightly.
- Weights SG-SAC improves tail tracking but loses tail reward by about `3.43%`.
- Markov SG-SAC improves tail reward and physical tracking slightly, but prediction-score and gain-drift diagnostics worsen slightly.
- All three detgate SG-SAC runs remain above the saved disturbed OF-MPC reward baseline.

## Validation

- Ran `C:\Users\hamediaa\.conda\envs\rl-env\python.exe report/scripts/analyze_polymer_sg_sac_detgate_hidden7_20260604.py`.
- Ran `C:\Users\hamediaa\.conda\envs\rl-env\python.exe -m py_compile report/scripts/analyze_polymer_sg_sac_detgate_hidden7_20260604.py`.
- Visually checked reward, policy-fraction, dominance, and physical tracking figures.
