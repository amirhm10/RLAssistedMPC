# Polymer Markov z_bound 0.50 Default

Date: 2026-06-09

## Summary

Increased the active polymer Markov standalone default `z_bound` from `0.20` to `0.50`. Polymer combined inherits the same Markov controller default, so its parity test now expects the same value.

## Local Evidence

Recent saved SG-TD3 Markov disturbed mismatch runs show that the wider `0.20` bound was the best of the recent small sweep:

| z_bound | Run | Tail-20 RL reward | Tail-20 OF-MPC reward | Tail delta |
| --- | --- | ---: | ---: | ---: |
| 0.05 | `Polymer/Results/sg_td3_markov_critic_warm3_ls_else_mpc_shadow_disturb_mismatch/20260608_155310` | -3.7572 | -3.8084 | 0.0512 |
| 0.10 | `Polymer/Results/sg_td3_markov_critic_warm3_ls_else_mpc_shadow_disturb_mismatch/20260608_193914` | -3.7485 | -3.8084 | 0.0598 |
| 0.20 | `Polymer/Results/sg_td3_markov_critic_warm3_ls_else_mpc_shadow_disturb_mismatch/20260609_013736` | -3.5404 | -3.8084 | 0.2680 |

The `0.20` run also used the larger radius rather than leaving it idle: post-warm executed `z` 2-norm q95 was about `0.3616`, and tail q95 was about `0.3923`, with a four-coordinate max of `0.4`.

## Interpretation

The `0.20` result suggests the polymer Markov SG-TD3 actor and supervisor gate can exploit more Markov authority than the earlier `0.05` and `0.10` settings. Moving to `0.50` is therefore a controlled next stress test of the same mechanism.

## Risk To Watch

Because Markov corrections modify the prediction matrix used by MPC, the wider bound can create distorted predictions if the critic gate releases poor actor actions. Watch post-warm worst reward, `z` norm q95/max, source fractions, and shadow projection activity.

## Verification

- Python syntax checks for the modified polymer defaults and tests.
- `tests/test_supervisor_gated_markov_integration.py`
- `tests/test_polymer_combined_runner.py`
