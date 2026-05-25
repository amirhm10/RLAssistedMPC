# 2026-05-25 Distillation TD3 BC-Handoff Without Release Gate

Implemented the diagnostic BC-handoff profile for active distillation TD3 families: weights, residual, and Markov.

Key behavior changes:

- Disabled the protected-BC release gate in the active TD3 defaults.
- Replaced the separate controlled-authority ramp with a BC-integrated raw-action handoff.
- The handoff blends from the safe action to TD3 over the first 10 subepisodes with authority `0.1 -> 1.0`.
- Kept hard bounds: actor raw action limits, weight/residual bounds, physical input headroom, Markov `z_bound = 0.04`, and Markov vector norm cap `0.06`.
- Disabled residual rho/beta authority for this diagnostic run, leaving only residual bounds and physical headroom.
- Disabled Markov TD3 priority fallback, reward probation, LS runtime fallback, and candidate gate vetoes by default.
- Kept Markov LS computation as a BC target and diagnostic signal.

Validation:

- Python compile passed for the changed config, shared BC helper, three TD3 runners, and active distillation TD3 entrypoints.
- Config/helper checks confirmed 10-subepisode BC, release gate disabled, handoff authority `0.1 -> 1.0`, old TD3 authority ramp inactive, residual authority disabled, and Markov priority fallback disabled.
- No full Aspen simulation was run.
