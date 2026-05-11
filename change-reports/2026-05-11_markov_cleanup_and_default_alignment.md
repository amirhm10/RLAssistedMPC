# 2026-05-11 Markov Cleanup And Default Alignment

- removed the temporary Step 3 probe notebook and probe runner after confirming the shared polymer disturbance fix
- restored the main Markov notebook to use the standard notebook parameter flow without a notebook-local warm-start override
- aligned `POLYMER_MARKOV_DEFAULTS["episode_defaults"]` with the standard polymer RL notebook defaults by reusing `POLYMER_MATRIX_DEFAULTS["episode_defaults"]`
- checked the normal Markov tuning values and left them unchanged because they were already at the expected baseline:
  - `z_bound = 0.05`
  - `band_floor_phys = [0.006, 0.07]`
