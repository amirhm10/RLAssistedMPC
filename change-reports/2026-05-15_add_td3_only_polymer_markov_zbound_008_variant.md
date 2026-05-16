## Summary

Added a new polymer Markov notebook variant, `RL_assisted_MPC_markov_zbound_008_looser_gate_only_td3_only_unified.ipynb`, cloned from the existing z = 0.08 looser-gate notebook but configured to execute TD3 actions without LS fallback.

## Files changed

- `RL_assisted_MPC_markov_zbound_008_looser_gate_only_td3_only_unified.ipynb`
- `utils/markov_runner.py`

## Notes

- The new notebook pins `rl_fallback_to_ls = False` and `force_td3_execute = True`.
- The shared Markov runner now lets `force_td3_execute` activate TD3 execution from step 0 instead of waiting for `step >= predict_h`.
