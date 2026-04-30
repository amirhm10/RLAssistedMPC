## Summary

- fixed distillation Aspen snapshot cleanup so the migrated unified notebooks resolve the real `AM_C2S_SS_simulation*` snapshot directories instead of the non-existent plain `C2S_SS_simulation*` folders
- moved Aspen snapshot deletion to run after `CloseDocument` and `Quit`, then broadened cleanup to remove both `.snp` and `.tsnp` snapshot files
- added workspace VS Code update settings to reduce automatic app and extension reload interruptions during long runs

## Files Changed

- `systems/distillation/config.py`
- `systems/distillation/plant.py`
- `.vscode/settings.json`

## Verification

- `C:\Users\HAMEDI\miniconda3\envs\rl-env\python.exe -c "from systems.distillation.config import resolve_aspen_paths; ..."` now resolves baseline snapshots to `...\\AM_C2S_SS_simulation2` and confirms the directory exists
- `C:\Users\HAMEDI\miniconda3\envs\rl-env\python.exe -m py_compile systems\\distillation\\config.py systems\\distillation\\plant.py`
