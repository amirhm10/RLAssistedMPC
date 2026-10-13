# Polymer Scenario 1 comparison runner

Author-selected scope: one seed (7), polymer only, the original **200-cycle** `legacy_gradual_200` scenario. The 300-cycle robustness profile belongs to the separate Scenario 2 campaign. Existing scientific modules, entrypoints, identification assets, and user-owned outputs are read-only to this wrapper.

## CMD commands

```cmd
call "C:\ProgramData\anaconda3\Scripts\activate.bat" rl-assisted-env
cd /d "C:\Users\hamediaa\OneDrive - McMaster University\PythonProjects\RL_assisted_MPC"
python run_polymer_comparisons.py --dry-run
python run_polymer_comparisons.py --smoke
```

Production LS-only:

```cmd
python run_polymer_comparisons.py --methods ls
```

After LS completes, run the other three comparators:

```cmd
python run_polymer_comparisons.py --methods arx bo threshold
```

The default command without `--methods` selects all four comparators. Each runs in an isolated worker, sequentially, with one BLAS/PyTorch CPU thread by default. Threshold comparison adds one combined critic-gated reference only when no completed matching production reference can be verified. Completed matching LS/SG jobs under this runner's Scenario 1 output root are automatically reused. Different source hashes, seeds, profiles, incomplete runs, or smoke runs are not equivalent references. Explicit references can be supplied with `--ls-reference` or `--sg-reference`, pointing to the job directory. Historical submissions without verified provenance are not silently accepted as same-seed references.

Only the specifically requested missing LS production run is launched during implementation. Other production studies are ready for the commands above. Current E1/Scenario 2 jobs are left running.

## Fixed scenario

- 200 cycles, two 400-step setpoints per cycle: `(eta,T) = (4.5 L/g,324 K)` and `(3.4 L/g,321 K)`; total 160,000 steps.
- Original `Ts=0.5 h`, input bounds, initial conditions, scaling assets, complete gradual `qi/qs/ha` arrays, original reward, SLSQP settings and unshifted zero initial optimizer guess.
- Conventional MPC uses `Np=9`, `Nc=3`, `Q0=[5,1]`, `R0=[1,1]`.
- RL arms preserve supervisor execution in cycles 1–10, critic-only handoff 11–13, original eligibility release in cycle 14, original learning through 199, deterministic evaluation in 200, and original inclusive step-boundary semantics. Networks, learning/replay schedules, horizon holds, execution order and downstream checks remain unchanged.
- Analytical LS/ARX adaptation remains active during evaluation as part of their adaptive controller, rather than being confused with neural training.

## Implemented methods

**LS:** the existing `run_markov_correction_supervisor` path with `run_rl_proposal=False`. No RL agent is instantiated. Its native bounded LS timing, regularizer, prediction improvement, drift, nominal-cost and solver acceptance checks remain unchanged. Original runner logs may retain their SG formatting; actual LS execution is source code 5 in `rl_action_source_log`. The exported method is explicitly `ls`, with `agent_kind=none`.

**ARX:** measured incremental histories replace the predictor's internal state for optimization. Each output has two autoregressive lags. Viscosity uses coolant lags 3–5 and monomer lags 1–2; temperature uses lags 1–2 for both inputs (13 fitted coefficients total). Initial coefficients are derived and checked against the existing nominal Markov sequence. RLS uses forgetting `.995`, covariance `1000 I`, minimum regressor squared norm `1e-12`, covariance eigenvalues clipped to `[1e-8,1e8]`, and retains the last finite predictor whose dynamic poles lie inside the unit circle. Updates begin after five measured samples. Predictions start at the current measured output and propagate increments; there is no extra disturbance integrator. Nominal observer calculations remain solely as diagnostics and are not inserted into the new ARX state. This changes both free and forced predictions, unlike LS's forced-response correction. No model-order/delay sweep, extra excitation or noise is introduced.

**BO:** tune `Q=[5 alpha,1]`, `R=[beta,beta]`, with independent multipliers in `[.75,2]`. Forty trials include nominal `(1,1)`, four corners and 35 GP expected-improvement choices. Use normalized log parameters, Matérn 5/2 covariance, nugget `1e-6`, two optimizer restarts and 4096 seeded Sobol candidates; EI offset is `.01 * max(std(observed costs),1e-12)`. Objective is negative original mean reward. A shared nominal 10-cycle prefix supplies a clean cloneable plant and observer state; every trial independently replays original cycles 11–12 (64,000 trial steps + 8000 prefix steps). The original long-ramp arrays are sliced, never regenerated as a two-cycle ramp. Failed trials get `C_nom + 100 max(1, abs(C_nom))` and cannot win; invalid nominal calibration stops the method. The selected valid weights are frozen for a fresh complete 200-cycle run. The repeated calibration portion is not held-out; final evaluation is never used to tune the weights.

**Threshold:** preserve the entire original combined RL training calculation, replacing only eligible learned-versus-supervisor acceptance. `E=max_i abs(y_i-r_i)/max(k_rel_i abs(r_i),band_floor_i)` uses current physical, before-action measurements and the existing reward bands. Calibrate `kappa=max(1,1.25*q95(E))` using a nominal 10-cycle prefix; then freeze it. Eligible finite proposals pass when `E<=kappa`, otherwise the existing supervisor action executes. Original acceptance behavior is retained inside the critic-only handoff so that hidden-proposal diagnostics and exploration state are preserved. Readiness, horizon holds, accepted LS-or-zero supervisor, action bounds, projections, executed-action replay and final neural freeze remain intact. Critics still train; DQN values still generate horizon proposals. Critic acceptance scores are computed as shadow diagnostics, so this implementation does not claim critic inference cost savings. Four channel decisions use the same reactive principle without a joint performance guarantee. This is an end-to-end training comparison, not a frozen-policy causal selector experiment.

## Outputs and analysis

New bundles are under `Polymer/Results/comparisons/scenario1/<timestamp_id>/`. Parent manifest records hashes, seed, profile, interpreter and git/dirty state; each method saves `job.json`, `status.json`, `run.log`, exact resolved configuration, `training.pkl`, `metrics.json` and timing. BO adds trial records; ARX saves coefficients and accepted updates; threshold saves calibration and per-channel decisions. RL arms export final CPU inference network state dictionaries (not complete training-resume checkpoints). Files are never silently overwritten; failed attempts remain visible.

Every production parent audits the saved nominal MPC bundle: full references/disturbances/scaling/initial state, physical input units, bounds, all-step reward recomputation, and an 800-step native nominal replay. Replay tolerances are **1% of minimum physical tracking bands** for outputs and **0.01% of actuator ranges** for inputs; raw differences are recorded. This audit establishes numerical compatibility, not bitwise replay of an older environment. The legacy `u_mpc` field is physical; its `delta_u_storage` is scaled. Referenced files are recorded rather than copied or overwritten.

```cmd
python run_polymer_comparisons.py --audit-baseline
python run_polymer_comparisons.py --analyse "C:\full\path\to\comparison_batch"
```

Analysis exports unsmoothed fixed reward, output-wise physical RMSE/MAE/max error/IAE and input movement for cycles 14–20, 14–200, 191–199, 196–200 and 200. Tracking uses `y[k+1]-reference[k]`, preserving after-action alignment. Online timing is inter-plant-step work excluding integration and the first setup interval; replay/reward/learning following one step count in the next interval. Wrapper and diagnostic overhead remain included. BO worker time includes calibration; its main controller timing is recorded separately. Solver-failure counts include all optimizer calls and are not physical-constraint certificates.

Smoke mode is explicitly technical: 200 cycles with a clock stride of 200 (800 total steps), unchanged controller horizons/architectures, and five BO trials. It exercises release/freeze, fitting, calibration and exports; its compressed dynamics/rewards are not manuscript performance data.

## Scientific background and claim limits

- Åström and Wittenmark (1973), [On Self-Tuning Regulators](https://doi.org/10.1016/0005-1098(73)90073-3): LS-based self-tuning foundation; their SISO minimum-variance formulation is not this MIMO ARX-MPC implementation.
- Forssell and Ljung (1999), [Closed-loop identification revisited](https://doi.org/10.1016/S0005-1098(99)00022-9): interpretation of closed-loop identification and its assumptions; no inherited convergence/unbiasedness claim.
- Sorourifar et al. (2021), [MPC tuning using constrained BO](https://doi.org/10.1016/j.ifacol.2021.08.249), and Hirt et al. (2024), [Learning MPC parameters using BO](https://doi.org/10.1016/j.ifacol.2024.08.426): outer-loop performance-driven MPC tuning; this two-weight deterministic comparator does not reproduce their complete model/constraint/uncertainty formulations.
- Lakhani et al. (2022), [Threshold-supervised RL-assisted PID tuning](https://doi.org/10.20517/ces.2021.15): related cumulative-error fallback; their PID architecture and stability proof do not apply to this instantaneous MPC threshold.

The self-contained mathematical protocol and verified bibliography are in the active paper revision folder as `polymer_scenario1_comparison_protocol.tex/.pdf` and `polymer_comparison_references.bib`. Original-source freeze hashes are in the dated comparison change report.

One realization supports this polymer case comparison, not statistical superiority or transfer of comparator performance to distillation. Predictor-pole rejection and nominal fallback are numerical/operational rules, not safety, stability, feasibility, or state/output constraint guarantees. E1 critic validation, paired Scenario 2 repeats, unseen references, joint-action analysis, stress/sensitivity studies and NMPC remain distinct evidence tasks. No completed reviewer response or performance claim follows from development/smoke verification alone.
