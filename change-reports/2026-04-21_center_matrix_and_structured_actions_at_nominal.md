# 2026-04-21 Center Matrix And Structured Actions At Nominal

- fixed matrix and structured multiplier mapping so raw action `0` maps to nominal multiplier `1.0` instead of the midpoint of asymmetric bounds
- added shared centered multiplier mapping helpers in `utils/multiplier_mapping.py`
- updated `utils/matrix_runner.py` to use the centered mapping
- updated `utils/structured_model_update.py` and `utils/structured_matrix_runner.py` call path to use the centered mapping
- updated the combined supervisor model branch so combined matrix-style actions use the same nominal-centered convention
- verified numerically that `action=-1` maps to the low bound, `action=0` maps to `1.0`, and `action=1` maps to the high bound
