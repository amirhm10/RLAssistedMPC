# Polymer Markov Legacy Mimic Ladder

Date: 2026-05-10

This report is the cumulative record for the isolated unified-to-legacy polymer Markov mimic ladder. The experiment surface is intentionally separate from the shared runner so each legacy-mimic delta can be tested without disturbing the other notebook families.

## Frozen references

- Frozen legacy target: `Polymer/Results/polymer_markov_corrected_mpc/20260510_204243/`
- Frozen unified reference: `Polymer/Results/td3_markov_disturb/20260510_204834/`
- Forked notebook: `polymer_markov_corrected_mpc_unified_legacy_mimic.ipynb`
- Forked runner: `utils/markov_runner_legacy_mimic.py`
- Step analysis script: `report/scripts/analyze_polymer_markov_legacy_mimic_ladder.py`

## Ladder definition

| Step | Code delta | Status |
| --- | --- | --- |
| Step 0 | Unified clone baseline | implemented |
| Step 1 | Runtime context parity | implemented |
| Step 2 | Nominal-reference solve parity | implemented |
| Step 3 | Plant-step / disturbance parity | implemented |
| Step 4 | TD3 construction parity | implemented |
| Step 5 | Comparator / reporting parity | implemented |
| Step 6 | Residual delta hunt | reserved |

## How to run

1. Open `polymer_markov_corrected_mpc_unified_legacy_mimic.ipynb`.
2. Set `LEGACY_MIMIC_STEP` to one of:
   - `step0_unified_clone`
   - `step1_runtime_context_parity`
   - `step2_nominal_reference_parity`
   - `step3_plant_step_parity`
   - `step4_td3_construction_parity`
   - `step5_comparator_reporting_parity`
   - `step6_residual_delta_hunt`
3. Run the notebook in disturbed polymer mode with the current 50-test setup.
4. Then run:
   `C:\Users\HAMEDI\miniconda3\python.exe report/scripts/analyze_polymer_markov_legacy_mimic_ladder.py --step <step_name>`
5. Paste the generated metrics and figure path into the matching section below.

## Acceptance criteria

- TD3 fraction within `+/-0.05` of frozen legacy
- LS fraction within `+/-0.05` of frozen legacy
- nominal fraction within `+/-0.02` of frozen legacy
- mean executed `||z||` within `+/-0.01` of frozen legacy
- mean gain drift within `+/-0.01` of frozen legacy
- output-1 RMSE to frozen legacy `<= 0.01`
- output-2 RMSE to frozen legacy `<= 0.03`

## Step 0

Code delta:
Unified clone baseline only.

Why this is a legacy difference:
This is not a legacy difference yet. It proves the fork reproduces current unified behavior before any mimic edits are activated.

Run used:
Pending first mimic run.

Did it move closer to legacy?
Pending.

Figures:
Pending.

## Step 1

Code delta:
Runtime context parity through the legacy bounds source from `system_data`.

Why this is a legacy difference:
The legacy script builds control bounds from identified system artifacts, while the unified notebook reconstructs them locally from notebook-level inputs.

Run used:
Pending.

Did it move closer to legacy?
Pending.

Figures:
Pending.

## Step 2

Code delta:
Legacy nominal lifted G0 solve path.

Why this is a legacy difference:
The candidate filter should compare against the same nominal action and nominal reference cost as the legacy script.

Run used:
Pending.

Did it move closer to legacy?
Pending.

Figures:
Pending.

## Step 3

Code delta:
Legacy-style polymer disturbance write and plant step inside the live loop.

Why this is a legacy difference:
The LS teacher and TD3 proposal are both path-dependent, so plant-step timing must match legacy exactly.

Run used:
Pending.

Did it move closer to legacy?
Pending.

Figures:
Pending.

## Step 4

Code delta:
Legacy-style TD3 seed semantics with deterministic default attribution mode.

Why this is a legacy difference:
The legacy script does not forward a dedicated seed into `TD3Agent(...)`, which changes network initialization and exploration history.

Run used:
Pending.

Did it move closer to legacy?
Pending.

Figures:
Pending.

## Step 5

Code delta:
Always save an internal nominal rerun bundle for legacy-style comparator analysis.

Why this is a legacy difference:
Legacy compares against its own internal nominal rerun, not only the canonical saved baseline bundle.

Run used:
Pending.

Did it move closer to legacy?
Pending.

Figures:
Pending.

## Step 6

Code delta:
Residual low-level delta hunt only if needed.

Why this is a legacy difference:
This step is reserved for any remaining causal difference that survives Steps 1 through 5.

Run used:
Pending.

Did it move closer to legacy?
Pending.

Figures:
Pending.
