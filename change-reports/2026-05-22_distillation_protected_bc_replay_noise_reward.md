# 2026-05-22 Distillation Protected-BC, Replay, Noise, And Reward Defaults

## Scope

Implemented the protected behavioral-cloning rollout profile for active distillation families: weights, residual, Markov, horizon, and dueling horizon. Matrix, structured-matrix, and reidentification families were intentionally left out of the new protected-BC behavior.

## Changes

- Distillation shared reward defaults now use `Q_diag = [37000, 5000]`.
- Distillation residual correction bounds now use `[-0.02, 0.02]` per input, with rho authority still controlled by the existing residual settings.
- Active distillation replay defaults now use `buffer_size = 50000`, `replay_frac_per = 0.4`, `replay_frac_recent = 0.3`, and `replay_recent_window_mult = 10` for TD3, SAC, DQN, and dueling DQN active-family defaults.
- Active continuous TD3 distillation defaults now use parameter noise with `param_noise_std_start = 0.2`, `param_noise_std_end = 0.02`, target policy smoothing `0.1`, and noise clip `0.2`.
- Distillation DQN/DDQN and dueling defaults keep `[512, 512, 512, 512, 512]`, NoisyNet exploration, and now pass `noisy_sigma_init = 0.5` explicitly from config.
- Added a shared protected-BC release gate helper that tracks raw action gap, max-coordinate gap, rolling gate status, release step, and blocked/released logs.
- Weight TD3 now trains during protected warm-up toward the identity multiplier action while execution stays at identity until the release gate passes.
- Residual TD3 now trains during protected warm-up toward the executed/projection-safe residual action and stores executed residual actions in replay.
- Markov TD3 now trains during protected warm-up toward `z_to_raw_action(safety_projected_ls_z)` and keeps live TD3 authority behind the protected release gate.

## Validation

- `py_compile` passed for the changed distillation config, runner, helper, and entrypoint files.
- Config checks confirmed active replay, TD3 noise, DQN NoisyNet, `Q_diag`, and residual bound defaults.
- Pure protected-BC helper checks confirmed warm-up training can be active, large action gaps block release, and sustained small gaps release live authority.
- Lightweight runner construction checks confirmed the Markov TD3 agent resolves to the new replay and exploration defaults without launching Aspen.

Full Aspen distillation simulations were not run because this change only needed static/config/path validation.
