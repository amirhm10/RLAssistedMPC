# Polymer Markov Unified Algorithm And Tuning Start

Date: 2026-05-11

This note is the new baseline algorithm document for future polymer Markov tuning runs. The active notebook surface is now [RL_assisted_MPC_markov_unified.ipynb](/c:/Users/HAMEDI/OneDrive%20-%20McMaster%20University/PythonProjects/RL_assisted_MPC/RL_assisted_MPC_markov_unified.ipynb), and the active shared runtime is the Step 3-consolidated Markov path in [markov_runner.py](/c:/Users/HAMEDI/OneDrive%20-%20McMaster%20University/PythonProjects/RL_assisted_MPC/utils/markov_runner.py) plus the shared polymer disturbance contract in [helpers.py](/c:/Users/HAMEDI/OneDrive%20-%20McMaster%20University/PythonProjects/RL_assisted_MPC/utils/helpers.py:305).

## Objective

The method keeps the offset-free polymer MPC structure, but adds a finite-horizon Markov correction layer identified online by least squares and proposed online by TD3. The corrected controller should:

- improve recent plant prediction quality
- preserve nominal MPC as the final fallback
- use the historical polymer disturbance semantics as the default live-step contract

## Plant And Offset-Free Model

The nonlinear polymer plant is the CSTR in `Simulation/system_functions.py`, with controlled outputs:

$$
y_k =
\begin{bmatrix}
\eta_k \\
T_k
\end{bmatrix},
$$

and manipulated inputs:

$$
u_k =
\begin{bmatrix}
Q_{c,k} \\
Q_{m,k}
\end{bmatrix}.
$$

The disturbed polymer live runs also vary plant-side quantities:

$$
d^{\mathrm{plant}}_k =
\begin{bmatrix}
Q_{i,k} \\
Q_{s,k} \\
hA_k
\end{bmatrix}.
$$

The nominal linear augmented model used by the offset-free controller is:

$$
x_{a,k+1} = A_{\mathrm{aug}} x_{a,k} + B_{\mathrm{aug}} \Delta u_k,
$$

$$
y_k = C_{\mathrm{aug}} x_{a,k},
$$

with observer update:

$$
\hat{x}_{a,k+1} = A_{\mathrm{aug}} \hat{x}_{a,k} + B_{\mathrm{aug}} \Delta u_k + L (y_k - \hat{y}_k).
$$

The live notebook uses scaled deviation coordinates around the steady state:

$$
\Delta u_k = u^{\mathrm{scaled}}_k - u^{\mathrm{scaled}}_{\mathrm{ss}},
\qquad
\Delta y_k = y^{\mathrm{scaled}}_k - y^{\mathrm{scaled}}_{\mathrm{ss}}.
$$

## Markov Correction Model

For prediction horizon $P$ and control horizon $M$, the nominal lifted prediction is:

$$
Y_k = Y^{\mathrm{free}}_k + G_0 \Delta U_k.
$$

The finite-horizon Markov blocks are:

$$
M_i = C_{\mathrm{aug}} A_{\mathrm{aug}}^{\,i-1} B_{\mathrm{aug}},
\qquad i = 1,\dots,P.
$$

The controller parameterizes a corrected block sequence as:

$$
M_i(z_k) = M_{i,0} + \sum_{j=1}^{r} z_{j,k} B^{(j)}_i,
$$

where $B^{(j)}_i$ are basis blocks and $z_k \in \mathbb{R}^r$ is the correction vector. The corrected lifted matrix is:

$$
G(z_k) = \mathrm{Toeplitz}\!\left(M_1(z_k), \dots, M_P(z_k)\right).
$$

The active notebook default uses the `io_pair_gain` basis family, so each $z_j$ scales one output-input Markov channel over the horizon.

## LS Identification And Prediction Score

The online LS candidate is fit over a recent prediction window by minimizing the corrected output error with a quadratic regularizer:

$$
z^{\mathrm{LS}}_k
=
\arg\min_z
\sum_{\tau \in \mathcal{W}_k}
\left\|
Y^{\mathrm{meas}}_\tau -
\left(
Y^{\mathrm{free}}_\tau + G(z)\Delta U^{\mathrm{real}}_\tau
\right)
\right\|_{W_y}^2
+
\lambda_z \|z\|_2^2.
$$

The main usefulness score is the prediction-improvement score:

$$
S_{\mathrm{pred}}(z_k)
=
\sum_{\tau \in \mathcal{W}_k}
\left(
\|W_y E_0(\tau)\|_2^2 - \|W_y E_z(\tau)\|_2^2
\right)
-
\lambda_z \|z_k\|_2^2,
$$

with

$$
E_0(\tau) = Y^{\mathrm{meas}}_\tau - \left(Y^{\mathrm{free}}_\tau + G_0 \Delta U^{\mathrm{real}}_\tau\right),
$$

$$
E_z(\tau) = Y^{\mathrm{meas}}_\tau - \left(Y^{\mathrm{free}}_\tau + G(z_k) \Delta U^{\mathrm{real}}_\tau\right).
$$

The live controller accepts a corrected candidate only if its score clears:

$$
S_{\mathrm{pred}}(z_k) > s_{\mathrm{pred,min}},
$$

and the corrected lifted model also passes a gain-drift guard.

## RL State, RL Action, And Selection Logic

The TD3 state is built from the current observer state, tracking error, innovation, previous input deviation, previous executed correction, LS correction, LS score, and LS gain drift. The TD3 action is a bounded raw action mapped into:

$$
z_k \in [-z_{\max}, z_{\max}]^r.
$$

The executed selection logic is:

1. Compute the nominal MPC move.
2. Compute the LS Markov candidate when enough history is available.
3. Let TD3 request a Markov correction.
4. Execute:

$$
\text{TD3} \rightarrow \text{LS} \rightarrow \text{nominal MPC}.
$$

The default notebook release schedule currently keeps:

- episodes 1-10: LS teacher only
- episode 11 onward: TD3, then LS fallback, then nominal fallback

Nominal MPC remains the final safety fallback whenever the corrected candidate does not pass the score, gain-drift, or feasibility guards.

## Step 3 Disturbance-Stepping Contract

The dominant implementation fix is now part of the algorithm definition.

For polymer disturbed live runs, the disturbance step at time $k$ is applied as plant attributes before the plant state is advanced:

$$
(Q_{i,k}, Q_{s,k}, hA_k) \longrightarrow \text{plant attributes} \longrightarrow \texttt{system.step()}.
$$

This is now the canonical shared contract across polymer disturbed runners. It matters because the Markov LS fit, TD3 replay, and candidate acceptance are all path-dependent. If disturbance timing is changed, the recent prediction window, executed trajectory, and replay distribution all change.

## Current Unified Defaults

The consolidated notebook currently assumes these main defaults:

| Knob | Current default |
| --- | --- |
| `run_mode` | `disturb` |
| `n_tests` | `50` |
| `warm_start` | notebook override `10` |
| `predict_h` | `9` |
| `cont_h` | `3` |
| `basis_family` | `io_pair_gain` |
| `z_bound` | `0.05` |
| `prediction_window` | `20` |
| `lambda_z` | `1.0e-3` |
| `s_pred_min` | `1.0e-6` |
| `gain_drift_max` | `0.10` |
| `nominal_solver_mode` | `lifted_g0_prototype` |
| `rl_fallback_to_ls` | `True` |
| `force_td3_execute` | `False` |
| `rl_store_executed_action_in_replay` | `True` |

## First Tuning Table

| Knob | Expected effect | Main metric to watch |
| --- | --- | --- |
| `warm_start` | Changes how much LS teacher data is injected before TD3 is trusted | action-source fractions, replay push quality |
| `z_bound` | Changes correction authority | gain drift, nominal fallback fraction, input movement |
| `prediction_window` | Changes how local or global the LS fit is | LS score, executed score, first-move difference |
| `lambda_z` | Shrinks aggressive corrections | full-sequence difference, gain drift, reward |
| `s_pred_min` | Tightens or loosens release of corrected candidates | TD3 accepted fraction, nominal fraction |
| `gain_drift_max` | Controls structural deviation from nominal lifted dynamics | gain drift, output RMSE to baseline |
| `nominal_cost_relative_tol` | Changes how conservative the cost guard is | LS fallback fraction, nominal fallback fraction |
| TD3 exploration settings | Changes policy diversity and replay coverage | requested score, executed score, reward trend |

## Immediate Tuning Workflow

Use this document as the starting point for the next iteration:

1. keep the Step 3 disturbance semantics fixed
2. tune release schedule and authority before changing the basis family
3. compare every run with:
   - action-source fractions
   - mean executed `||z||`
   - mean gain drift
   - first-move difference
   - full-sequence difference
   - nominal-cost margin
4. only revisit basis design after release scheduling and score thresholds are stable

## Files Used

- [RL_assisted_MPC_markov_unified.ipynb](/c:/Users/HAMEDI/OneDrive%20-%20McMaster%20University/PythonProjects/RL_assisted_MPC/RL_assisted_MPC_markov_unified.ipynb)
- [markov_runner.py](/c:/Users/HAMEDI/OneDrive%20-%20McMaster%20University/PythonProjects/RL_assisted_MPC/utils/markov_runner.py)
- [helpers.py](/c:/Users/HAMEDI/OneDrive%20-%20McMaster%20University/PythonProjects/RL_assisted_MPC/utils/helpers.py)
- [notebook_params.py](/c:/Users/HAMEDI/OneDrive%20-%20McMaster%20University/PythonProjects/RL_assisted_MPC/systems/polymer/notebook_params.py)
