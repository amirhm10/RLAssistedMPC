# Raise Replay Buffer Defaults

Date: 2026-04-24

## Summary

- Raised the shared polymer RL replay-buffer default from `40_000` to `150_000` transitions.
- Raised the shared distillation RL replay-buffer default from `40_000` to `150_000` transitions.
- Raised the polymer poles experiment `buffer_capacity` default to `150_000` for consistency with polymer RL notebooks.

## Notes

- The active unified notebooks read these defaults through `get_polymer_notebook_defaults(...)` and `get_distillation_notebook_defaults(...)`.
- Existing dirty warm-start edits in the same parameter files were left untouched.
