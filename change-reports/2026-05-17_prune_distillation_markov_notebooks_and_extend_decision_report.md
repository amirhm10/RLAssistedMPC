# 2026-05-17 prune distillation Markov notebooks and extend decision report

## What changed

- removed these distillation Markov notebook entrypoints:
  - `distillation_RL_assisted_MPC_markov_relaxed_acceptance_unified.ipynb`
  - `distillation_RL_assisted_MPC_markov_ls_only_unified.ipynb`
  - `distillation_RL_assisted_MPC_markov_td3_without_ls_unified.ipynb`
- kept only:
  - `distillation_RL_assisted_MPC_markov_unified.ipynb`
  - `distillation_RL_assisted_MPC_markov_td3_only_no_safeguard_unified.ipynb`
- extended `report/distillation_markov_td3_decision_authority_2026_05_17.md`

## Why

The saved May 16 analysis shows the extra variant notebooks were useful for diagnosis, but not as long-term active experiment surfaces.

The active notebook surface is now reduced to:

1. one conservative reference Markov notebook
2. one TD3-priority no-safeguard notebook

The report was also expanded to document:

- why the older gate logic suppresses TD3
- how to implement TD3-first fallback step by step
- how to think about LS-target behavioral cloning after warm start
- why replay should still store executed action when fallback changes the plant move
- why executed-action replay becomes more acceptable once TD3 has real live authority

## Scope

- notebook entrypoint cleanup only
- no raw result folders were deleted
- no runner/controller code was changed in this task

