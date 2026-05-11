# 2026-05-11 Fix Polymer Markov Disturbance Attribute Mapping

- updated `utils/helpers.py` so dict-based polymer disturbance schedules map lowercase helper keys `qi`, `qs`, and `ha` onto the actual `PolymerCSTR` plant attributes `Qi`, `Qs`, and `hA`
- this restores the shared disturbed live-step path to the verified Step 3 semantics used by the historical polymer Markov legacy-mimic run
- scope is intentionally minimal: shared disturbance profiles can still be stored with lowercase keys, while the plant-step helper now writes the correct live plant attributes before `system.step()`
