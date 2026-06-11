# Combined Markov/Residual Parameter-Noise Guard

Date: 2026-06-10

## Change

- Polymer combined now requires Markov and residual TD3 configs to use `exploration_mode = "param_noise"`.
- Distillation combined now applies the same Markov/residual parameter-noise guard.
- Both combined entrypoints now save an `agent_config_snapshot` inside `combined_cfg`, so future combined `input_data.pkl` bundles record the horizon, Markov, weights, and residual agent configs used to construct the agents.

## Rationale

The latest polymer standalone Markov and residual checks support using parameter noise for high-authority Markov/residual agents. Combined runs already deep-copy the standalone Markov and residual TD3 defaults; the new guard makes that dependency explicit and prevents accidental drift back to Gaussian action noise.

## Validation

- Static config validation is expected to confirm combined Markov and residual configs resolve to `param_noise 0.10 -> 0.02`.
- No polymer or Aspen closed-loop simulation is launched by this change.
