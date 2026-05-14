## 2026-05-14 Polymer Markov looser-acceptance analysis

### Goal

Analyze the latest polymer Markov wider-range looser-acceptance run after the user observed that TD3 acceptance was almost zero even though the notebook was made looser.

### What was added

- report script:
  `report/scripts/generate_polymer_markov_looser_acceptance_assets_20260514.py`
- report note:
  `report/polymer_markov_looser_acceptance_2026_05_14.md`
- figure bundle:
  `report/figures/polymer_markov_looser_acceptance_20260514/`

### Main finding

The looser score and nominal-cost settings were not the active bottleneck. The fixed gain-drift guard dominated after `z_bound` was widened from `0.08` to `0.40`, so TD3 requested candidates passed the score and cost checks often enough but failed the drift check almost always.

### Why it matters

This means the latest polymer Markov run should be interpreted as a stronger LS/nominal gated controller rather than a successful TD3-authority result. The slight reward improvement is therefore not evidence of stronger TD3 online execution.
