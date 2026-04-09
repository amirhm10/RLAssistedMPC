# Replay Buffer Cleanup And Agent Fix Plan

## Goal

Clean up the replay-buffer and replay-related agent logic in the current `RLAssistedMPC` project so it matches the actual online-training workflow used in this repository.

This plan intentionally removes legacy functionality inherited from the previous project and keeps only the parts relevant to the current continuous-process online RL setting.

The target outcome is:
- one mixed replay buffer for TD3 and SAC
- replay behavior controlled from notebook/config defaults
- continuous-task handling preserved with `done = 0.0` during normal operation
- correct twin-critic priority updates for TD3 and SAC
- end-of-rollout multistep flushing
- removal of all legacy pretraining code paths

## Scope

Apply these changes to the active repo code used by the current unified notebooks. Focus on:
- `TD3Agent/agent.py`
- `SACAgent/sac_agent.py`
- `TD3Agent/replay_buffer.py`
- any notebook parameter/config files for polymer and distillation
- any runners affected by end-of-rollout multistep flushing
- any helper code that still exposes old pretraining or old buffer-mode switching

Do not add new replay algorithms. Do not add a second replay buffer. Do not preserve the old pretraining workflow.

## High-level decisions already made

### 1. Keep only the mixed replay buffer for continuous agents

For TD3 and SAC, the project should use only the mixed replay buffer.

The plain replay buffer should no longer remain as an algorithm choice for the current project workflow. If shared storage code is useful internally, Codex may keep a small internal base-storage/helper abstraction, but from the user/project perspective there should be no online choice between:
- plain replay buffer
- mixed replay buffer

The intended behavior is:
- TD3 always uses the mixed replay buffer
- SAC always uses the mixed replay buffer

### 2. Replay knobs must be notebook/config controlled

The current hidden replay defaults inside the buffer implementation must be moved out to the notebook/config layer so they behave like the other run parameters.

Expose at least these knobs in notebook/config defaults:
- `buffer_size`
- `replay_frac_per`
- `replay_frac_recent`
- `replay_recent_window_mult`
- `replay_recent_window`
- `replay_alpha`
- `replay_beta_start`
- `replay_beta_end`
- `replay_beta_steps`

The notebooks/configs should own the defaults.
The resolved values should appear in the run summary/printout just like other important parameters.

### 3. New default replay-capacity policy

Reduce the default replay capacity to:
- `buffer_size = 40000`

This is intentional and should become the new default for the current project unless an individual notebook overrides it.

### 4. Recent window must depend on setpoint length

Remove the fixed hardcoded recent-window assumption and derive it from the notebook run settings.

Use:

`recent_window = min(buffer_size, replay_recent_window_mult * set_points_len)`

Use the default:
- `replay_recent_window_mult = 5`

This keeps distillation behavior close to the current implicit setting while making polymer scale more sensibly.

### 5. Continuous-task semantics stay intact

The process is continuous.
Normal operation should keep:
- `done = 0.0`

Do **not** fake terminal `done = 1.0` at setpoint changes.
Do **not** force logical cuts at setpoint switches just because the setpoint changes.

Since the setpoint is part of the state, multistep targets are allowed to cross setpoint changes in this project.

### 6. The only required multistep task-boundary fix

The one required multistep fix is:
- flush the n-step accumulator at the end of each rollout/run

Reason:
if `done` is never true during ordinary operation, the last `n - 1` transitions can be lost unless the accumulator is flushed at the end.

This end-of-rollout flush is required for multistep use.

### 7. Fix replay priorities for TD3 and SAC

TD3 and SAC both use twin critics in this codebase.
Replay priorities should not be based only on `q1`.

Replace single-critic priority updates with a twin-critic aggregate priority.

Use this default rule:

`priority = 0.5 * (abs(y - q1) + abs(y - q2))`

Apply this in:
- TD3 online training
- SAC online training
- SAC PER update paths that still exist before cleanup
- any TD3 legacy update path that remains before cleanup

DQN does not need this particular twin-critic fix.
Do not change DQN priority logic for this reason.

### 8. Remove legacy pretraining completely

This repo no longer needs the old pretraining workflow inherited from the previous project.
Remove it completely.

That includes removing:
- `pretrain_push(...)`
- `pretrain_from_buffer(...)`
- `pretrain_add(...)`
- any old `mode == "mpc"` or equivalent replay-mode branch used only for pretraining
- any config/defaults/docs/comments related only to that old pretraining workflow

Keep normal checkpoint save/load behavior.
Do not remove ordinary online replay/training.

## Items explicitly removed from scope

Do **not** include the following in the implementation plan:

1. Do not implement full mixture-distribution importance weighting for recent/uniform/PER samples.
   The current heuristic weighting is acceptable for now.

2. Do not implement a two-buffer replay design.
   No separate recent buffer and anchor buffer.

3. Do not add forced logical segmentation at setpoint switches.

4. Do not add new replay algorithms such as HER, reservoir replay, or additional buffer families.

## Detailed implementation tasks

### Task A. Simplify TD3 replay-path design

Update `TD3Agent/agent.py` so that:
- TD3 always uses the mixed replay buffer for the current project
- legacy buffer-mode switching tied to pretraining is removed
- there is no user-facing plain-buffer-vs-mixed-buffer branch for TD3
- all replay-related hyperparameters are accepted from config/notebook inputs

Things to remove or refactor:
- the old `mode` logic that selects replay-buffer type for pretraining reasons
- any dead code that exists only because pretraining once existed
- any replay API that is no longer used after pretraining removal

### Task B. Simplify SAC replay-path design

Update `SACAgent/sac_agent.py` so that:
- SAC always uses the mixed replay buffer for the current project
- `use_per` or similar legacy toggles are removed if they only preserve an old alternate buffer path
- replay-related hyperparameters come from notebook/config inputs

If an internal minimal abstraction is useful, Codex may keep one, but externally the project should reflect one current replay design, not multiple competing replay modes.

### Task C. Refactor the mixed replay buffer API

Update `TD3Agent/replay_buffer.py` so that the mixed replay buffer:
- receives its replay knobs through constructor arguments instead of relying on hidden defaults inside `sample()` and `sample_sequence()`
- stores those values on the object
- uses them consistently in both one-step and sequence sampling

The target constructor should support at least:
- `capacity`
- `state_dim`
- `action_dim`
- `default_discount`
- `alpha`
- `beta_start`
- `beta_end`
- `beta_steps`
- `frac_per`
- `frac_recent`
- `recent_window`
- `eps`

Update internal sampling so these values come from object attributes unless explicitly overridden for an experiment.

### Task D. Add twin-critic replay priority updates

In TD3 and SAC:
- compute priority from both critics instead of only `q1`
- use the average absolute TD error by default:

`priority = 0.5 * (abs(y - q1) + abs(y - q2))`

Be careful to preserve tensor shapes and detach correctly before priority update.

### Task E. Add end-of-rollout n-step flush

Wherever TD3/SAC runners or rollout code end a run/rollout, add logic so that if multistep replay is active:
- the agent flushes the n-step accumulator at the end of the rollout
- all remaining partial transitions are pushed into replay

This should not change continuous-task semantics inside the rollout.
During normal operation, `done` stays `0.0`.

Do not add fake terminal flags at setpoint switches.
Only ensure end-of-rollout flush happens reliably.

If needed, add a small public method on TD3/SAC such as:
- `flush_nstep()`

and call it from the appropriate runner teardown/end-of-rollout location.

### Task F. Remove all legacy pretraining hooks

Search the repo and remove legacy pretraining functionality related to the previous project.
This includes code, dead config parameters, stale comments, and unused branches.

At minimum, verify removal of:
- `pretrain_push`
- `pretrain_from_buffer`
- `pretrain_add`
- old replay-mode branches used only for pretraining
- notebook/config fields that only exist to support those paths

After cleanup, the code should clearly represent the current workflow:
- online training only
- mixed replay only for TD3/SAC

### Task G. Move replay defaults into notebook/config files

Update the active notebook parameter/config files for polymer and distillation so replay knobs are visible and editable like the other run parameters.

Add defaults such as:
- `buffer_size = 40000`
- `replay_frac_per = 0.5`
- `replay_frac_recent = 0.2`
- `replay_recent_window_mult = 5`
- `replay_alpha = 0.6`
- `replay_beta_start = 0.4`
- `replay_beta_end = 1.0`
- `replay_beta_steps = 50000`

Use those defaults unless a notebook explicitly overrides them.

Also update resolved summary printouts so the user sees replay settings clearly.

### Task H. Derive `recent_window` from notebook run settings

In the notebook/runtime assembly code:
- compute `recent_window` from `set_points_len`
- apply the formula:

`recent_window = min(buffer_size, replay_recent_window_mult * set_points_len)`

Pass the resolved value into agent/buffer construction.

### Task I. Preserve DQN behavior on the twin-critic issue

Do not modify DQN or Dueling DQN priority logic for the twin-critic issue.
That fix applies only to TD3 and SAC.

## Validation checklist

Codex should not stop at editing. It should also verify the cleanup.

### Functional checks

1. TD3 and SAC instantiate with only the mixed replay path.
2. Replay knobs are visible in notebook/config defaults.
3. The resolved run summary prints the replay settings.
4. `buffer_size` default is `40000` unless overridden.
5. `recent_window` is derived from `set_points_len` and capped by `buffer_size`.
6. TD3 priority updates use both critics.
7. SAC priority updates use both critics.
8. DQN behavior remains unchanged for this specific issue.
9. End-of-rollout multistep flush occurs and does not break continuous-task semantics.
10. Legacy pretraining functions and mode branches are removed.

### Code-quality checks

1. No stale references remain to deleted pretraining functions.
2. No stale notebook/config options remain that refer to removed pretraining paths.
3. TD3/SAC constructors and replay-buffer constructors have clean, explicit arguments.
4. The code remains consistent with current online use only.

## Deliverables

Codex should produce:

1. The code changes implementing this plan.
2. A short summary of what was changed, grouped by:
   - replay buffer
   - TD3
   - SAC
   - notebook/config defaults
   - runner/end-of-rollout handling
   - legacy pretraining removal
3. A note listing any places where old dead code was removed.
4. A note confirming that no new replay algorithm was introduced.

## Final implementation intent

This is a cleanup-and-correction pass, not a method-expansion pass.

The expected final design is:
- continuous-task online RL
- one mixed replay buffer for TD3/SAC
- notebook-controlled replay behavior
- smaller, more adaptation-focused default replay capacity
- derived recent window from setpoint length
- correct twin-critic priority updates
- end-of-rollout multistep flush
- no legacy pretraining code
