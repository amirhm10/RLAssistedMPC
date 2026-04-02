# 2026-04-01 rl-env Default Environment

## Summary

- Registered the `rl-env` Jupyter kernel with display name `Python (rl-env)`.
- Updated repo notebook metadata to point at `rl-env` instead of the generic `python3` kernel.
- Updated `AGENTS.md` so future work uses `rl-env` as the default notebook/runtime environment.

## Important Boundary

- Several notebooks already had unrelated local edits before this change.
- To avoid committing in-progress experiment work, only the clean notebook metadata updates should be committed.
- The already-dirty notebooks can keep the `rl-env` metadata locally without being swept into this commit.
