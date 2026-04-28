# 2026-04-22 Extend Wide-Range Report With Latest Polymer And Distillation Reward Analysis

- Updated [report/generate_polymer_wide_range_report.py](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/report/generate_polymer_wide_range_report.py) to incorporate the latest polymer matrix and structured-matrix capped reruns.
- The refreshed report now distinguishes the first capped reruns from the latest capped reruns, which shows that the latest structured polymer run improved only after the intended tight-`A`, wide-`B` cap was actually applied.
- Added cross-system reward-geometry analysis for polymer and distillation, including a reward-balance sweep figure and new in-report tables explaining edge-balance versus bonus-balance targets.
- Added a distillation-safe-wide-search section with a gated wide-search / backtracking-acceptance design concept for preventing catastrophic performance collapse while preserving access to a wide raw multiplier space.
- Updated the observer-refresh section to answer the practical question directly: every-episode observer-pole redesign is likely too aggressive, and slower thresholded refresh is more plausible than unconditional episode-by-episode redesign.
