# 2026-05-05 Ignore Polymer Five-Seed Report Artifacts

- Added a `.gitignore` rule for generated polymer five-seed report asset folders under `report/figures/`.
- Removed the tracked polymer five-seed figure, CSV, JSON, and markdown outputs from git so future pushes keep the report source without bundling generated study artifacts.
- Kept the local files in place; this change only affects repository tracking.
