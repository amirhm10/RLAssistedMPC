## Polymer Markov Replay-Storage A/B Setup

- Added a notebook-level override `RL_STORE_EXECUTED_IN_REPLAY_OVERRIDE` to `RL_assisted_MPC_markov_unified.ipynb` so polymer Markov can be run with either executed-action replay or requested-action replay while keeping the rest of the setup fixed.
- When the override is used, the notebook now auto-tags the result and compare prefixes with `replay_exec` or `replay_req` so the two A/B runs are easier to distinguish.
- Updated `utils/markov_runner.py` so the saved `input_data.pkl` bundle records both `rl_store_executed_action_in_replay` and a readable `replay_storage_mode`.
- Added `report/scripts/analyze_polymer_markov_replay_storage_ab.py` to compare the two replay-storage variants on:
  - average reward by episode
  - TD3 accepted fraction by episode
  - requested TD3 score by episode
  - LS score by episode
  - `||z_TD3 - z_LS||` by episode
