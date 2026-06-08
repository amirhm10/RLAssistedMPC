# Polymer Combined SG/Plain Runner Modes

## Summary

Restored `RL_assisted_MPC_combined_unified.py` as an active polymer entrypoint using the archived combined runner as the base.

The active combined runner now has one top-level mode:

- `combined_agent_mode="sg"`: SG-DQN horizon plus SG-TD3 Markov, weights, and residual agents.
- `combined_agent_mode="plain"`: DQN horizon plus plain TD3 Markov, weights, and residual agents.

The default is SG mode with mismatch-state inputs.

## Implementation Notes

- Polymer combined defaults now align with the current active single-runner recipes: warm start `10`, horizon action freeze `3`, TD3 action/actor freeze `3`, residual rho/deadband live authority disabled, and Markov live safety disabled with shadow diagnostics retained in config.
- The root combined runner uses only DQN/SG-DQN and TD3/SG-TD3. SAC, dueling DQN, TD7, and matrix agents remain outside the active combined path.
- `utils.agent_step_runtime` now includes SG-TD3 continuous selection/replay helpers, and `utils.combined_runner` uses them for combined Markov, weights, and residual branches when the agent kind is `sg_td3`.
- SG replay records executed action plus policy action, supervisor action, previous action, selected source, scores, and advantage.

## Validation Scope

Added focused combined tests for default config, SG/plain kind resolution, root runner source guardrails, hidden-window supervisor execution, and supervised continuous replay metadata. Full polymer combined closed-loop training remains the acceptance run.
