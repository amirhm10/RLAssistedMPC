## 2026-05-14 Polymer Markov z = 0.08 looser-gate-only variant

### Goal

Create the clean follow-up polymer Markov notebook that separates gate loosening from authority widening.

### Notebook added

- `RL_assisted_MPC_markov_zbound_008_looser_gate_only_unified.ipynb`

### Variant settings

The new notebook starts from `RL_assisted_MPC_markov_unified.ipynb` and applies notebook-local overrides:

- `z_bound = 0.08`
- `nominal_cost_relative_tol = 0.30`
- `s_pred_min = 1.0e-7`

This keeps the reference authority level while loosening only the score and nominal-cost gates. The gain-drift guard remains unchanged.

### Output prefixes

Dedicated result directories were assigned so the run stays isolated from the earlier `z = 0.08` and `z = 0.40` experiments:

- result prefix: `td3_markov_disturb_zbound_008_looser_gate_only`
- compare prefix: `disturb_compare_td3_markov_zbound_008_looser_gate_only`

### Validation

- Cleared notebook outputs.
- Revalidated the notebook structure after editing.
- Ran a setup-cell sanity check to confirm the resolved runtime settings without launching the full training loop.
