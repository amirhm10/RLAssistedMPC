## Summary

Extended the residual workflow so the saved `input_data.pkl` includes a replay-buffer snapshot for later analysis.

## Current State

`xhatdhat` was already present in the residual result bundle, and because the plotting/export path saves the full bundle through `build_storage_bundle(...)`, it was already being written to `input_data.pkl`.

Replay-buffer contents were not being saved.

## Change

Added `export_snapshot(...)` methods to the active replay buffers in `TD3Agent/replay_buffer.py` and wired the residual runner to save:

- `replay_buffer_snapshot`

into the residual result bundle when the agent exposes a replay buffer with that export method.

The saved replay snapshot includes:

- buffer metadata
- `states`
- `actions`
- `rewards`
- `next_states`
- `dones`
- `discounts`
- `n_actual`
- `episode_ids`
- PER/recent metadata such as `priorities` and `birth_step` for the mixed replay buffer

The exported arrays are stored in chronological ring order for analysis.

## Validation

- `python -m py_compile TD3Agent/replay_buffer.py utils/residual_runner.py`
- direct `PERRecentReplayBuffer.export_snapshot(...)` smoke check under `rl-env`
- verified `build_storage_bundle(...)` preserves both `xhatdhat` and `replay_buffer_snapshot`
