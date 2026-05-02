## Summary

Updated the distillation matrix-study defaults to keep the guarded Step 2 / warm-start freeze path and switch TD3 exploration from parameter noise to conservative Gaussian action noise.

## Details

- Distillation scalar matrix defaults now document the protected Step 2 release cap and post-warm-start action/actor freeze as the intended guarded path.
- Distillation TD3 matrix defaults now use `exploration_mode = "gaussian"` instead of `param_noise`.
- The TD3 scalar-matrix noise settings remain conservative:
  - `target_policy_smoothing_noise_std = 0.01`
  - `std_start = 0.01`
  - `std_end = 0.01`
- Distillation structured matrix defaults inherit the same TD3 agent settings from the scalar matrix defaults.
