# Safe RL Implementation Summary

Date: April 25, 2026

This document summarizes the current safe reinforcement learning implementation for the polymer CSTR control project. It is intended as a research-summary draft, so it emphasizes the method, implementation progress, and rationale behind the architecture. The current design combines a TD3 reinforcement learning controller with a Lyapunov-based safety layer and an MPC fallback controller. The central idea is to let RL propose high-performance control moves while retaining a model-based safety mechanism that checks each proposed action before it reaches the plant.

## Executive Summary

The safe RL controller is organized as a supervisory control stack. At each control step, the measured or estimated plant state is first converted into the scaled deviation coordinates used by the MPC and RL components. A target selector then computes an admissible local steady target, denoted by `x_s` for the physical state target and `u_s` for the input target. The TD3 policy proposes an action in the same deviation-coordinate input space. Before this action is applied, a Lyapunov safety layer evaluates whether the proposed input respects input bounds, input-rate bounds, and a one-step Lyapunov contraction condition relative to the selected target. If the RL action passes this check, it is applied directly. If the action does not pass, the supervisor falls back to a Lyapunov/tracking MPC action.

The project therefore follows a model-based safety-filter interpretation of safe RL. The RL policy is not trusted as the only source of constraint satisfaction. Instead, RL is treated as an upstream economic or performance-oriented controller, and the safety filter is the final authority on whether the proposed action is admissible for the current state and target. This is consistent with the broader safe RL literature, which separates performance optimization from constraint satisfaction and often introduces external safety mechanisms when learning controllers are deployed on constrained dynamical systems [1], [2], [3].

The implementation is currently built around the following modules:

- `Simulation/run_rl_lyapunov.py`: safe RL rollout supervisor, TD3 action generation, target selection, Lyapunov safety-filter call, fallback routing, and logging.
- `TD3Agent/agent.py`: TD3 actor-critic implementation, target networks, delayed policy updates, action noise, replay buffer interface, and training logic.
- `Lyapunov/target_selector.py`: refined Step A target selector that computes `x_s`, `u_s`, disturbance target `d_s`, and output target quantities.
- `Lyapunov/safety_filter.py`: Lyapunov safety filter for RL candidate actions, including candidate post-check and MPC fallback hooks.
- `Lyapunov/lyapunov_core.py`: Lyapunov value, contraction bound, candidate-action evaluation, and tracking MPC solver ingredients.
- `Lyapunov/direct_lyapunov_mpc.py`: direct Lyapunov MPC path used as a reference/fallback-style controller and as a diagnostic implementation for target and contraction behavior.

The implemented architecture can be summarized as:

```text
current observer state xhat,dhat
        |
        v
target selector: compute x_s, u_s, d_s, y_s
        |
        v
TD3 policy: propose u_RL
        |
        v
Lyapunov safety layer:
  accept u_RL if bounds and contraction hold
  otherwise call Lyapunov/tracking MPC fallback
        |
        v
apply safe input to polymer CSTR
```

## Control Problem and Coordinate System

The plant is a polymer continuous stirred-tank reactor with two manipulated inputs, typically represented as `Qc` and `Qm`, and output variables including viscosity-like `eta` and reactor temperature `T`. The control problem is not simply to track a setpoint in raw physical units. The implementation uses scaled and deviation-coordinate representations inherited from the baseline offset-free MPC and RL code. This is important because the safety filter, target selector, and RL policy must all agree on the representation of state, input, and output quantities.

The augmented observer state is denoted in the code as `xhatdhat` or `xhat_aug`. It contains the physical state estimate and an output-disturbance estimate. In the Lyapunov target-selector path, the augmented state is assumed to be ordered as `[x; d]`, where `x` is the physical model state and `d` is the output disturbance estimate. The target selector freezes the current disturbance estimate and computes a compatible steady target under the linear augmented model. The target package contains:

- `x_s`: selected physical steady-state target.
- `u_s`: selected input target in deviation/scaled coordinates.
- `d_s`: selected or frozen disturbance target.
- `y_s`: output at the selected target.
- `r_s` or `yc_s`: controlled-output reference associated with the selected target.
- diagnostic terms such as target residuals, bound margins, and objective components.

The target is recomputed at each control step because the observer estimate, setpoint, previous input, and previous target can all change. This converts the raw setpoint-tracking problem into a local tracking problem around an admissible model-consistent steady target. That design is central to the safe RL implementation: the Lyapunov check is evaluated relative to `x_s` and `u_s`, not relative to the raw setpoint directly.

## Step 1: Target Selector Computes `x_s` and `u_s`

The target selector is the first active model-based component in the safe RL loop. Its job is to translate the desired output setpoint into a feasible local equilibrium target for the augmented plant model. The refined Step A selector in `Lyapunov/target_selector.py` solves a convex steady-target problem. In simplified notation, the selector chooses `x_s` and `u_s` to minimize a weighted objective that includes output-target tracking, input anchoring, input smoothing, state smoothing, and anchoring to the current state estimate:

```text
minimize
    ||r_s - y_sp||_Qr^2
  + ||u_s - u_applied||_Ru_ref^2
  + ||u_s - u_s_prev||_R_delta_u^2
  + ||x_s - x_s_prev||_Q_delta_x^2
  + ||x_s - xhat||_Q_x_ref^2

subject to
    x_s = A x_s + B u_s + B_d d_hat
    u_min <= u_s <= u_max
    optional output bounds
```

The implementation also supports tightened input and output bounds, previous-target warm starts, selectable term activation, and diagnostic reporting of each objective term. The target selector returns a structured target package rather than only returning the numerical pair `(x_s, u_s)`. This matters for research reporting because it gives visibility into target feasibility, setpoint compromise, bound margins, dynamic residuals, and how much the selected target differs from the raw requested setpoint.

This target-selection layer is closely related to steady-state target optimization in offset-free MPC and tracking MPC. Classical MPC references emphasize the finite-horizon receding-horizon optimization structure, constraints, terminal ingredients, and target tracking design [4], [5]. In this project, the selector provides the steady target required by both the Lyapunov safety filter and the Lyapunov MPC fallback.

## Step 2: TD3 Proposes the Candidate Action

After target selection, the RL policy proposes a candidate input. The RL controller uses a TD3-style deterministic actor-critic structure implemented in `TD3Agent/agent.py`. TD3 is appropriate for this project because the manipulated inputs are continuous and because the algorithm was designed to reduce overestimation bias in actor-critic methods through clipped double Q-learning, delayed policy updates, target networks, and target policy smoothing [6].

In the rollout, the RL state is built from the observer state, current setpoint, and previous input deviation using the project scaling utilities. The actor produces an action in the learned action space, and the rollout converts that action into the deviation-coordinate input candidate `u_RL`. This proposed action is not immediately applied to the plant. It is first passed to the Lyapunov safety layer along with:

- the current augmented observer state `xhat_aug`;
- the current target package from the selector;
- model ingredients needed for Lyapunov prediction;
- input bounds and input-rate bounds;
- the previous input;
- fallback MPC configuration.

This separates performance learning from safety verification. The TD3 actor can learn fast, aggressive, or economically useful behavior, but the final applied input is still determined by the safety supervisor. In the code, this separation is visible in the handoff from `u_rl_dev` to `apply_lyapunov_safety_filter(...)` in `Simulation/run_rl_lyapunov.py`.

## Step 3: Lyapunov Check as the Safety Layer

The Lyapunov safety check evaluates the proposed action relative to the selected target. Let the physical state-estimation error be:

```text
e_x(k) = xhat(k) - x_s
```

Given a candidate input `u`, the one-step predicted error is:

```text
e_x(k+1) = A e_x(k) + B (u - u_s)
```

The Lyapunov value is:

```text
V(k) = e_x(k)' P_x e_x(k)
```

The safety layer accepts the candidate only if the next-step Lyapunov value satisfies the contraction-style bound:

```text
V(k+1) <= rho V(k) + eps_lyap
```

where `0 < rho < 1` and `eps_lyap >= 0`. The small `eps_lyap` term gives a numerical tolerance and can also represent practical convergence rather than exact one-step decrease to zero. The check also requires the candidate to satisfy input bounds and move bounds:

```text
u_min <= u <= u_max
du_min <= u - u_prev <= du_max
```

This post-check is implemented in `evaluate_candidate_action(...)` in `Lyapunov/lyapunov_core.py`. The safety filter stores diagnostic fields such as `V_k`, `V_next_cand`, `V_bound`, `candidate_bounds_ok`, `candidate_move_ok`, `candidate_lyap_ok`, and `lyap_margin`. These diagnostics are useful because they connect the applied control decision to a quantitative safety certificate at every time step.

The Lyapunov condition follows the stabilizing MPC tradition, where terminal costs, invariant/terminal sets, and contraction or decrease conditions are used to establish closed-loop stability properties under constraints [5]. The current project uses the Lyapunov condition as an online action admissibility test for the RL proposal, rather than relying on the RL training objective to discover safety by reward shaping alone.

## Step 4: Fallback to Lyapunov MPC

When the RL candidate is not admissible, the supervisor falls back directly to a model predictive controller. In the intended safe RL rollout, the fallback action is computed using the current model state, the selected tracking target, input constraints, move constraints, and the current MPC initial guess.

This fallback is important for a research-safe architecture because it gives the system a model-based controller to rely on whenever the RL action is outside the certified one-step safe set. Conceptually, the stack is:

```text
1. Prefer the RL action if it is certified.
2. Otherwise, use the Lyapunov/tracking MPC action.
```

This preserves the role of RL as the performance-oriented controller while maintaining a structured recovery path based on MPC. The fallback MPC is not a separate experiment bolted onto the controller; it is part of the online supervisor. It uses the same target-selector output, so the safety layer and fallback MPC agree on the local target pair `(x_s, u_s)`.

The fallback concept is also aligned with constrained policy-optimization and safe RL literature, where constraints are treated as first-class requirements rather than only as penalties in the reward function [1], [2]. The distinction in this project is that safety is enforced at deployment time using process-model and Lyapunov information, which is natural for a control system with an available linearized/identified model and input constraints.

## Current Implementation Progress

The implemented workflow now has the major pieces needed for a safe RL research prototype:

- A TD3 actor-critic controller exists for continuous-input control.
- The RL rollout can generate the RL state from the observer state, setpoint, and previous input.
- A refined target selector computes `x_s`, `u_s`, `d_s`, `y_s`, and `r_s` at each control step.
- The target selector includes target tracking, input anchoring, previous-target smoothing, state smoothing, and current-state anchoring terms.
- The target selector reports solver status, objective terms, target residuals, bound margins, and warm-start metadata.
- The Lyapunov core evaluates one-step candidate safety using a positive-semidefinite Lyapunov matrix `P_x`.
- The safety layer checks input bounds, move bounds, and Lyapunov contraction before applying an RL action.
- The supervisor carries forward the last valid target as a backup target when configured.
- The supervisor has an MPC fallback path that can be called when the RL action is not directly acceptable.
- Detailed diagnostics are stored for each step, including target information, candidate action information, final applied action, Lyapunov margins, fallback metadata, and solver metadata.

The result is a layered implementation rather than a single monolithic controller. This is useful for research writing because each layer has a clear purpose: target selection defines the local admissible reference, TD3 proposes a performance action, the Lyapunov layer certifies the action, and MPC provides the backup control law.

## Research Method Framing

The method can be described as a Lyapunov-filtered TD3 controller with MPC fallback for constrained polymer CSTR tracking. The following phrasing is suitable for a later research summary:

> At each sampling instant, a steady-state target selector computes an admissible local target pair `(x_s, u_s)` under the augmented process model and current disturbance estimate. A TD3 actor then proposes a continuous control action. Before plant application, the action is evaluated by a Lyapunov safety filter using a one-step contraction condition around `(x_s, u_s)`, together with input and move constraints. If the action satisfies the safety test, it is applied directly. Otherwise, the supervisor applies a Lyapunov/tracking MPC fallback action. This structure allows the learned controller to improve performance while preserving a model-based safety layer at deployment.

The key technical contribution of this implementation is the integration of target selection, TD3 action proposal, Lyapunov certification, and MPC fallback in a single rollout loop. Rather than training an RL controller and later evaluating whether it happened to remain safe, the online supervisor checks every candidate action before application.

## Relation to Literature

The project builds on several established ideas:

- Safe RL: Safe RL formalizes the need to optimize reward while respecting safety requirements during learning and deployment [1]. Constrained policy optimization frames constraints as explicit design objects rather than only reward penalties [2].
- Predictive safety filters: Predictive safety filters motivate wrapping learning-based controllers with a model-based safety layer that evaluates proposed inputs before they reach the plant [3].
- MPC and Lyapunov MPC: MPC provides constrained finite-horizon control with repeated online optimization, while stabilizing MPC theory uses Lyapunov/terminal ingredients to reason about stability and recursive behavior [4], [5].
- TD3 for continuous control: TD3 is a deterministic actor-critic method designed for continuous action spaces and reduces overestimation effects using twin critics, delayed actor updates, and target policy smoothing [6].

The current controller stack combines these ideas in a process-control setting. TD3 is used as the learned action generator, MPC supplies model-based control authority, and the Lyapunov safety filter provides the online action-certification mechanism.

## Implementation-Level Algorithm

The rollout can be written as the following high-level algorithm:

```text
Given observer state xhat_aug(k), previous input u(k-1), and setpoint y_sp(k):

1. Build the RL state from xhat_aug(k), y_sp(k), and u(k-1).

2. Solve target-selection problem:
       target_info(k) = selector(xhat_aug(k), y_sp(k), u(k-1))
       extract x_s(k), u_s(k), d_s(k), y_s(k)

3. TD3 actor proposes candidate:
       u_RL(k) = pi_theta(rl_state(k))

4. Candidate Lyapunov post-check:
       e_x(k) = xhat(k) - x_s(k)
       e_x(k+1) = A e_x(k) + B (u_RL(k) - u_s(k))
       V(k) = e_x(k)' P_x e_x(k)
       V_RL(k+1) = e_x(k+1)' P_x e_x(k+1)

   Accept u_RL(k) if:
       input bounds hold,
       move bounds hold,
       V_RL(k+1) <= rho V(k) + eps_lyap.

5. If not accepted, call Lyapunov/tracking MPC fallback.

6. Apply the selected safe input to the plant.

7. Store diagnostics for research analysis and update RL replay/training data.
```

## Notation

| Symbol | Meaning |
|---|---|
| `xhat_aug` | Augmented observer state `[xhat; dhat]` |
| `x_s` | Selected physical-state target |
| `u_s` | Selected steady input target |
| `d_s` | Selected/frozen output-disturbance target |
| `y_s` | Output predicted at the selected target |
| `u_RL` | Candidate input proposed by the TD3 actor |
| `u_safe` | Final input after safety filtering or fallback |
| `P_x` | Lyapunov matrix for physical-state error |
| `rho` | Lyapunov contraction factor |
| `eps_lyap` | Practical Lyapunov tolerance |
| `V_k` | Current Lyapunov value |
| `V_next` | One-step predicted Lyapunov value |

## References

[1] J. Garcia and F. Fernandez, "A Comprehensive Survey on Safe Reinforcement Learning," Journal of Machine Learning Research, vol. 16, pp. 1437-1480, 2015. https://jmlr.org/papers/v16/garcia15a.html

[2] J. Achiam, D. Held, A. Tamar, and P. Abbeel, "Constrained Policy Optimization," Proceedings of the 34th International Conference on Machine Learning, PMLR 70:22-31, 2017. https://proceedings.mlr.press/v70/achiam17a.html

[3] K. P. Wabersich and M. N. Zeilinger, "A Predictive Safety Filter for Learning-Based Control of Constrained Nonlinear Dynamical Systems," Automatica, vol. 129, article 109597, 2021. https://doi.org/10.1016/j.automatica.2021.109597

[4] J. B. Rawlings, D. Q. Mayne, and M. M. Diehl, Model Predictive Control: Theory, Computation, and Design, 2nd ed., Nob Hill Publishing, 2017. https://sites.engineering.ucsb.edu/~jbraw/mpc/

[5] D. Q. Mayne, J. B. Rawlings, C. V. Rao, and P. O. M. Scokaert, "Constrained Model Predictive Control: Stability and Optimality," Automatica, vol. 36, no. 6, pp. 789-814, 2000. https://doi.org/10.1016/S0005-1098(99)00214-9

[6] S. Fujimoto, H. van Hoof, and D. Meger, "Addressing Function Approximation Error in Actor-Critic Methods," Proceedings of the 35th International Conference on Machine Learning, PMLR 80:1587-1596, 2018. https://proceedings.mlr.press/v80/fujimoto18a.html
