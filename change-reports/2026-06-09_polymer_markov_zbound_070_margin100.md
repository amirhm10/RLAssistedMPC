# Polymer Markov z_bound 0.70 With SG Margin 1.0

Date: 2026-06-09

## Summary

Updated the active polymer Markov SG-TD3 standalone defaults for the next run:

- `controller.z_bound`: `0.50` -> `0.70`
- `supervisor_gate.advantage_margin`: `0.0` -> `1.0`

Polymer combined inherits the active Markov controller and Markov SG gate defaults, so the combined parity assertions were updated too.

## Rationale

The latest local z-bound sweep showed that `z_bound = 0.50` produced the best tail reward among `0.05`, `0.10`, `0.20`, and `0.50`:

- Tail-20 reward delta versus OF-MPC improved to about `+0.7378`.
- Tail executed `z` 2-norm q95 reached about `0.9866`, close to the four-coordinate vector cap for `z_bound = 0.50`.
- Worst post-warm reward worsened to about `-5.7549`.

The `0.70` run tests whether additional Markov authority can improve the tail further. Raising `advantage_margin` to `1.0` makes the SG-TD3 actor beat the supervisor by a larger critic-score margin before execution, which should make the first post-warm release more conservative.

## What To Watch

- Tail-20 reward delta versus OF-MPC.
- Worst first-20 post-warm reward.
- Tail executed `z` 2-norm q95 and max.
- SG policy fraction after warm start and in the tail.
- Shadow z-safety projection fraction.

If `0.70` improves tail reward but still worsens release, the next safer design is a post-warm z-bound ramp rather than another range expansion.

## Verification

- Python syntax checks for modified polymer defaults and tests.
- `tests/test_supervisor_gated_markov_integration.py`
- `tests/test_polymer_combined_runner.py`
