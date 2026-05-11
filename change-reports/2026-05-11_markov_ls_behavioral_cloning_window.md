# 2026-05-11 Markov LS Behavioral Cloning Window

- added a short post-warm-start behavioral-cloning window to the shared Markov workflow
- the Markov default now uses the existing behavioral-cloning schedule helper with:
  - `target_mode = "ls_action"`
  - `lambda_bc_start = 0.2`
  - `active_subepisodes = 5`
- the shared Markov runner now trains TD3 against the accepted LS raw action when LS is valid
- when LS is unavailable on a step, the behavioral-cloning target falls back to the actually executed safe action so the penalty does not fight accepted safe execution
- the Markov notebook now passes the behavioral-cloning config through to the shared runner and prints it in the resolved parameter summary
