## 2026-05-13 Polymer Markov wider-range looser-acceptance variant

### Goal

Create a separate polymer Markov unified notebook to test whether a much wider Markov correction range and looser acceptance thresholds improve closed-loop performance.

### Notebook added

- `RL_assisted_MPC_markov_wider_range_looser_acceptance_unified.ipynb`

### Variant settings

The new notebook starts from `RL_assisted_MPC_markov_unified.ipynb` and applies notebook-local overrides:

- `z_bound = 0.40`
- `nominal_cost_relative_tol = 0.30`
- `s_pred_min = 1.0e-7`

It also assigns dedicated output prefixes so the run will not overwrite existing polymer Markov results:

- result prefix: `td3_markov_disturb_zbound_040_looser_accept`
- compare prefix: `disturb_compare_td3_markov_zbound_040_looser_accept`

### Notes

- The canonical polymer Markov notebook was left untouched because it is already a live experiment surface.
- Outputs were cleared in the new notebook and the notebook JSON was revalidated after editing.
