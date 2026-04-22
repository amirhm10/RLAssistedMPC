# 2026-04-22 Polymer Matrix And Structured Range Widening

- widened the polymer default multiplier range from `0.75-1.25` to `0.6-1.3`
- kept the existing polymer `alpha` cap in place, so the effective polymer matrix defaults are:
  - plain matrix: `alpha in [0.6, 1.0566]`, `delta in [0.6, 1.3]`
  - structured matrix: `A in [0.6, 1.0566]`, `B in [0.6, 1.3]`
- left the generic structured `"wide"` profile unchanged because the polymer structured notebooks already apply explicit A/B override bounds from `systems/polymer/notebook_params.py`
