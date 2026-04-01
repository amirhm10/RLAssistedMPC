# Residual Methods

## Scope

This report explains the residual-control notebook family that adds a clipped
RL residual move on top of baseline MPC under model mismatch.

## Notebooks Included

- `RL_assisted_MPC_residual_model_mismatch.ipynb`
- `RL_assisted_MPC_residual_model_mismatch1.ipynb`
- `RL_assisted_MPC_residual_model_mismatch2.ipynb`
- `RL_assisted_MPC_residual_model_mismatch_multi.ipynb`

## Inputs And Outputs

- Input state: base scaled RL state plus innovation and tracking-error terms
- Action: residual move proposal in scaled delta-input space
- Output behavior: safe residual correction around the MPC baseline move

## Code Paths Explained

- notebook-local mismatch-band helpers
- notebook-local residual clipping and replay logic
- shared scaling helper usage from `utils/helpers.py`

## Same-Page Caveats

- The executed clipped residual is what gets replayed.
- This family relies heavily on notebook globals.
- Additional mismatch scheduling includes `CMf` in the main residual notebook.
