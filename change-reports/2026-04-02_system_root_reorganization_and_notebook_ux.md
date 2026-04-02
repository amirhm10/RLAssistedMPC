# 2026-04-02 System Root Reorganization And Notebook UX

## Summary
- Reorganized canonical assets into `Polymer/` and `Distillation/`.
- Moved active data dependencies to `Polymer/Data` and `Distillation/Data`.
- Routed active plot/result outputs to `Polymer/Results` and `Distillation/Results`.
- Removed the old root `Data`, `Result`, and `Results` directories after validation.

## Shared Layer Changes
- Added `systems/polymer` with canonical config, labels, and data-io helpers.
- Updated `systems/distillation` so its canonical data/result paths now point to `Distillation/Data` and `Distillation/Results`.
- Added `utils/notebook_setup.py` so notebooks resolve repo root, system directories, and distillation Aspen paths consistently.
- Updated `Simulation/sys_ids.py` so polymer system-ID experiments can save CSV outputs into an explicit data directory.

## Notebook UX Changes
- Updated all active polymer and distillation notebooks to expose system-specific data/result directories at the top.
- Added more visible run configuration comments and summary prints for selected paths and major settings.
- Added distillation Aspen preset selection with manual path override support.
- Switched active notebook save paths from the old root folders to the new system-specific result folders.

## Validation
- Verified shared loader smoke tests after the directory migration and root-folder deletion.
- Verified all active unified notebooks plus polymer `systemIdentification.ipynb` compile and pass `nbformat` validation after the rewrite.
