# Polymer Markov z_bound 0.70 Margin 0.0 Ablation

Date: 2026-06-09

## Summary

Changed the active polymer Markov SG-TD3 standalone default to keep the wider Markov authority while removing the stricter SG margin:

- `controller.z_bound`: remains `0.70`
- `supervisor_gate.advantage_margin`: `1.0` -> `0.0`

Polymer combined inherits the active standalone Markov gate and z-bound defaults, so its parity test was updated too.

## Rationale

The `z_bound = 0.70`, `advantage_margin = 1.0` run improved tail reward but sharply reduced policy passes:

- Tail reward delta versus OF-MPC: about `+0.9043`
- Post-warm SG policy pass fraction: about `6.9%`
- Tail SG policy pass fraction: about `13.4%`
- Worst post-warm reward: about `-7.7115`

This ablation tests whether keeping `z_bound = 0.70` but restoring `advantage_margin = 0.0` recovers more TD3 actor authority and clarifies whether the post-warm dip is mainly caused by the wider Markov range or by the stricter gate.

## Verification

- Python syntax checks for modified polymer defaults and tests.
- `tests/test_supervisor_gated_markov_integration.py`
- `tests/test_polymer_combined_runner.py`
