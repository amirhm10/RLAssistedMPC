# Final Parameter-Noise Ablation Defaults

Date: 2026-06-10

## Change

- Distillation Markov SG-TD3 now starts parameter-space exploration at `param_noise_std_start = 0.05` and keeps `param_noise_std_end = 0.02`.
- Polymer Markov SG-TD3 now uses `exploration_mode = "param_noise"` with `param_noise_std_start = 0.10` and `param_noise_std_end = 0.02`.
- Polymer residual SG-TD3 now uses `exploration_mode = "param_noise"` with `param_noise_std_start = 0.10` and `param_noise_std_end = 0.02`.

## Runtime Consistency

- Polymer standalone residual and polymer combined continuous-agent construction now pass `param_noise_std_start` and `param_noise_std_end` into `TD3Agent`/`SupervisorGatedTD3Agent`.
- Polymer combined continues to deep-copy the standalone Markov and residual TD3 configs, so the final-ablation exploration settings propagate into combined runs.

## Rationale

The final ablation makes exploration temporally coherent through parameter noise while reducing high-authority perturbations:

- distillation Markov: softer start than `0.10` to reduce post-warm release risk;
- polymer Markov/residual: `0.10` parameter noise instead of `0.20` Gaussian action noise to reduce stepwise actuator jitter while keeping persistent policy-level exploration.
