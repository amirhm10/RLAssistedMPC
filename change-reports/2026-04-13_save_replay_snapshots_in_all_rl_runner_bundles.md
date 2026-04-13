## Summary

Extended the active RL runner stack so replay-buffer contents are saved into `input_data.pkl` across the shared notebook families, not just the residual method.

## Scope

Updated shared runners for:

- horizon
- dueling horizon
- matrix
- structured matrix
- weights
- residual
- reid batch
- reid batch v2 / v3 (through the same runner)
- combined

## Change

Added replay-buffer snapshot export support to:

- `DQN/replay_buffer.py`
- `TD3Agent/replay_buffer.py`

Added a shared helper:

- `utils/replay_snapshot.py`

Single-agent runners now save:

- `replay_buffer_snapshot`

Combined now saves:

- `replay_buffer_snapshots`

with one entry per active sub-agent.

## Saved Data

The snapshots are exported in chronological ring order and include:

- buffer metadata
- `states`
- `actions`
- `rewards`
- `next_states`
- `dones`
- `discounts`
- `n_actual`
- `episode_ids`

and, for PER/recent buffers:

- `priorities`
- `birth_step`
- replay hyperparameters

Discrete replay snapshots additionally include:

- `behavior_prob`
- `behavior_logprob`

## Validation

- `python -m py_compile` on the touched replay and runner modules
- smoke-checked snapshot export for both DQN and TD3 replay buffers under `rl-env`
- verified `build_storage_bundle(...)` preserves both `replay_buffer_snapshot` and `replay_buffer_snapshots`
