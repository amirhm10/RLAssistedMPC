## 2026-05-13 Polymer Markov wider-range looser-acceptance variant

### Goal

Create a separate polymer Markov unified notebook to test whether a modestly wider Markov correction range and slightly looser acceptance thresholds improve closed-loop performance.

### Notebook added

- `RL_assisted_MPC_markov_wider_range_looser_acceptance_unified.ipynb`

### Variant settings

The new notebook starts from `RL_assisted_MPC_markov_unified.ipynb` and applies notebook-local overrides:

- `z_bound = 0.10`
- `nominal_cost_relative_tol = 0.15`
- `s_pred_min = 5.0e-7`

It also assigns dedicated output prefixes so the run will not overwrite existing polymer Markov results:

- result prefix: `td3_markov_disturb_zbound_010_loose_accept`
- compare prefix: `disturb_compare_td3_markov_zbound_010_loose_accept`

### Notes

- The canonical polymer Markov notebook was left untouched because it is already a live experiment surface.
- Outputs were cleared in the new notebook and the notebook JSON was revalidated after editing.
