# RL-Assisted MPC Design Summary For Polymer And Distillation

Date: 2026-04-25

## Executive Summary

This project studies **RL-assisted model predictive control (MPC)** for two process-control case studies:

- a polymer CSTR, and
- a distillation column.

The main design idea is not to replace MPC. Instead, MPC remains the deterministic optimization layer that enforces the local control structure, prediction horizon, input constraints, and offset-free tracking behavior. Reinforcement learning (RL) is used as a supervisory layer around MPC. The RL agents adjust selected MPC design variables, such as prediction-model multipliers, structured matrix multipliers, residual input corrections, weights, or horizons. This gives the controller adaptive authority while preserving the interpretability and constraint-handling strengths of MPC.

The approach was successful in both case studies, but in different ways. In the polymer case, RL-assisted MPC produced clear reward improvements over baseline MPC for the matrix and structured-matrix families, especially after release protection and diagnostic cap analysis were added. In the distillation case, the framework was also successful as a design and diagnosis method: it exposed that the distillation column is much more sensitive to model-multiplier authority, showed that residual policies can produce physically meaningful improvements under quadratic scoring, and identified why some reward-shaped results degrade even when the plant behavior is partly better. The distillation matrix policy is not yet ready for direct transfer without protection, but the RL-assisted MPC workflow successfully found the bottleneck: moderate `B`-matrix changes can be harmful even when `A` is tightly capped.

The final architecture should be understood as a layered control design:

| Layer | Role |
|---|---|
| Offset-free MPC | Baseline constrained controller and nominal safe fallback. |
| RL supervisor | Adapts model, weights, residual inputs, or horizons. |
| Offline diagnostics | Tests spectral radius, finite-horizon gain, and multiplier sensitivity before training. |
| Release protection | Temporarily limits authority during the first live policy release. |
| Acceptance or shadow gates | Check whether RL decisions are safe and potentially useful before transfer to harder systems. |
| Reward sensitivity analysis | Separates true plant improvement from reward-shaping artifacts. |

The project therefore demonstrates a practical RL-assisted MPC workflow: use RL for adaptation, keep MPC as the optimizer, and use diagnostics and release guards to avoid unsafe authority jumps.

## Background And Literature Context

MPC is a natural baseline for process systems because it solves a constrained finite-horizon optimization problem at each control step. Modern MPC design emphasizes prediction models, constraints, stability, observer design, and numerical optimization, all of which are central in process control applications [Rawlings et al., 2017](https://sites.engineering.ucsb.edu/~jbraw/mpc/MPC-book-2nd-edition-5th-printing.pdf).

RL is attractive because it can learn supervisory policies from closed-loop experience. Classical deep RL results showed how value-function learning and function approximation can solve high-dimensional sequential decision problems [Mnih et al., 2015](https://www.nature.com/articles/nature14236). For continuous control, actor-critic methods such as DDPG extended deep RL to continuous actions [Lillicrap et al., 2015](https://arxiv.org/abs/1509.02971). TD3 later improved deterministic actor-critic learning by reducing overestimation through clipped double critics and delayed policy updates [Fujimoto et al., 2018](https://arxiv.org/abs/1802.09477). SAC added a maximum-entropy stochastic actor-critic framework that improves exploration and stability in continuous-control tasks [Haarnoja et al., 2018](https://arxiv.org/abs/1801.01290).

The control setting here is close to the broader literature on learning for real physical systems: RL can discover behavior that is hard to hand-design, but the control problem must manage safety, sample efficiency, constraints, and robustness [Kober et al., 2013](https://doi.org/10.1177/0278364913495721). The project follows that principle. RL is not allowed to directly choose unconstrained physical inputs. It supervises an MPC layer, and the learned decisions are filtered by authority limits, diagnostics, and fallback logic.

## Baseline MPC Formulation

The baseline controller is an offset-free linear MPC built from system-identification data and scaling artifacts. At each control step, the controller solves a finite-horizon optimization problem using a nominal augmented model:

$$ x_{k+1} = A_0 x_k + B_0 \Delta u_k. $$

The output prediction is:

$$ y_k = C x_k. $$

The typical MPC cost penalizes tracking error and input movement:

$$ J = \sum_{i=1}^{N_p} \|y_{k+i} - y_{k+i}^{sp}\|_Q^2 + \sum_{i=0}^{N_c-1} \|\Delta u_{k+i}\|_R^2. $$

This baseline MPC is the reference controller for all comparisons. In both polymer and distillation, the RL supervisors are evaluated by comparing their closed-loop reward, physical tracking error, and input movement against this baseline.

## RL Supervisory Families

The repository contains several RL-assisted MPC families. They share a common principle: the RL action modifies the controller, not the plant input directly.

### Matrix Multiplier Supervisor

The scalar matrix multiplier family uses a low-dimensional action to scale the nominal prediction model:

$$ A_{\theta} = \alpha A_0, \qquad B_{\theta} = B_0 \operatorname{diag}(\beta_1,\beta_2). $$

For the augmented offset-free model, the physical dynamic block is scaled while the disturbance/integrator structure is preserved. The action is low-dimensional, interpretable, and easy to diagnose:

- `alpha` changes model memory/dynamics,
- `B_col_1` changes the first manipulated-input gain,
- `B_col_2` changes the second manipulated-input gain.

This method worked well in polymer after the authority release problem was addressed. It was more difficult in distillation because even tight `A` and moderate `B` changes can create poor MPC moves.

### Structured Matrix Supervisor

The structured-matrix family generalizes scalar multipliers by giving separate authority to blocks or bands of the `A` matrix and selected `B` columns:

$$ A_{\theta} = \mathcal{S}_A(A_0,\theta_A), \qquad B_{\theta} = \mathcal{S}_B(B_0,\theta_B). $$

This allows the actor to change different state substructures differently. The benefit is richer model adaptation. The risk is stronger coupling: each coordinate can look safe individually, but a joint candidate can still be poor. That is why per-coordinate diagnostics and later candidate-benefit diagnostics are important.

### Residual Supervisor

The residual family keeps the nominal MPC input and adds a learned correction:

$$ u_t^{RL} = u_t^{MPC} + \Delta u_t^{res}. $$

For distillation, this family was particularly informative. The residual policy did not catastrophically crash like the matrix policy. It showed physically meaningful improvements under pure quadratic scoring, especially in temperature and input movement. However, the current shaped reward penalizes the residual result because x24 composition tracking worsens in the tail. The residual method is therefore successful as a control-design direction, but it needs projection-aware learning so that the actor learns within the executable authority envelope.

### Weights And Horizons

The project also includes weight and horizon supervisors:

- weight agents adjust MPC penalty weights,
- horizon agents adjust prediction and control horizons.

These families are useful because they change controller aggressiveness without directly changing the plant model. They are often safer than broad model adaptation, but they may have less authority to overcome model mismatch.

## Training Algorithms

The main continuous-action algorithms are TD3 and SAC. TD3 is the current primary workhorse for matrix, structured-matrix, weight, and residual policies. It is appropriate because the action spaces are continuous and low-dimensional. TD3 uses:

- an actor network for the deterministic policy,
- two critic networks,
- delayed actor updates,
- target policy smoothing,
- replay-buffer training.

SAC is also present as an alternative stochastic actor-critic algorithm. SAC can be useful when exploration and entropy matter, but the current active matrix and residual results mainly use TD3.

DQN is used for discrete horizon-selection experiments. That is appropriate because horizon choices are naturally discrete.

## Safety And Diagnostic Layers

The largest lesson from the project is that RL authority must be released carefully. A policy that performs well after recovery can still damage early live episodes. The current workflow uses three diagnostic stages.

### Step 1: Offline Rho And Gain Sensitivity

Step 1 evaluates candidate multipliers before training or before live release. It computes:

- spectral radius of candidate physical dynamics,
- finite-horizon Markov gain,
- local log-space sensitivity by coordinate,
- random candidate scans,
- diagnostic suggested bounds.

The spectral-radius check is:

$$ \rho(A_{\theta}) = \max_i |\lambda_i(A_{\theta})|. $$

The finite-horizon gain check builds a Markov matrix from `A`, `B`, and `C` over the prediction horizon. This detects candidates that may be stable but still produce too much input-output gain. This was useful for polymer and essential for understanding why distillation is sensitive.

### Step 2: Release-Protected Advisory Caps

Step 2 uses Step 1 diagnostic bounds only during first live release. The actor still trains in the wide action space, but the executed multipliers are clipped during protected release and then ramped back to full authority:

$$ \log(\theta_{\mathrm{eff}}) = (1-r)\log(\theta_{\mathrm{diag}}) + r\log(\theta_{\mathrm{wide}}). $$

This solved a major polymer problem. The actor initially requested aggressive multipliers, but Step 2 prevented the plant from seeing the most damaging release actions. In polymer, Step 2 preserved the later full-authority improvement while reducing the first-live crash.

### Step 3: Acceptance And Shadow Gates

Step 3 originally tested whether a candidate MPC sequence was no worse than nominal MPC under the nominal objective:

$$ J_t^{\mathrm{cand|nom}} \leq J_t^{\mathrm{nom}} + \epsilon_{\mathrm{abs}}. $$

That strict version behaved almost exactly like MPC because nominal MPC is already the optimizer of the nominal objective. A relaxed version used:

$$ J_t^{\mathrm{cand|nom}} \leq (1+\epsilon_{\mathrm{rel}})J_t^{\mathrm{nom}} + \epsilon_{\mathrm{abs}}. $$

The relaxed version gave authority back, but it did not reliably improve polymer performance. This showed that nominal-cost closeness is a safety test, not a performance test. The next preferred design is a shadow dual-cost diagnostic:

$$ \mathrm{nominal\_penalty}_t = J(U_t^{cand};A_0,B_0) - J(U_t^{nom};A_0,B_0). $$

$$ \mathrm{candidate\_advantage}_t = J(U_t^{nom};A_t^{cand},B_t^{cand}) - J(U_t^{cand};A_t^{cand},B_t^{cand}). $$

The candidate should eventually be trusted only when it is safe enough under the nominal model and useful enough under its own candidate model.

## Polymer Case Study

The polymer CSTR case is the clearest success case for RL-assisted MPC. The plant has two controlled outputs:

- viscosity `eta`,
- reactor temperature `T`.

The manipulated inputs are:

- coolant flow `Qc`,
- monomer flow `Qm`.

The matrix and structured-matrix methods were able to improve over baseline MPC when the authority release problem was handled. Step 1 identified sensitive multiplier directions. Step 2 then converted the diagnostic bounds into a temporary release guard. This allowed the actor to keep learning wide-range model corrections while preventing the first live release from damaging the plant.

The polymer results support three conclusions:

1. RL-assisted MPC can outperform fixed nominal MPC when the model multiplier action is expressive enough.
2. The first live release must be protected; otherwise, the actor can request extreme model changes before the critic has learned accurate values.
3. Structured matrix authority is useful but needs stronger diagnostics because joint model changes can be risky even when individual coordinates are inside diagnostic bounds.

Therefore, polymer demonstrates a successful RL-assisted MPC design: RL can adapt MPC prediction behavior and improve closed-loop reward, while the MPC layer retains the optimization structure.

## Distillation Case Study

The distillation column has two controlled outputs:

- tray-24 ethane composition,
- tray-85 temperature.

The manipulated inputs are:

- reflux flow,
- reboiler duty.

Distillation is harder than polymer. The latest matrix run, before Step 2 or Step 3 were active on distillation, had tight `A` authority and wide `B` authority with low exploration and policy smoothing noise. It still degraded heavily. That does not mean the framework failed. It means distillation is more sensitive to model-gain changes, especially `B`-matrix directions.

The residual distillation result is more promising. Under the current shaped reward, residual loses in the final score because tail composition tracking worsens. But under pure quadratic and no-bonus reward candidates, the residual trajectory can score better than MPC. It improves temperature tracking and reduces input movement, but it sacrifices enough composition accuracy that the current bonus-shaped reward rejects it.

This is a successful outcome for design because it identifies the next control question precisely:

- the residual method can produce useful physical behavior,
- the reward currently emphasizes composition-band success strongly,
- the residual policy often asks for corrections that the `rho` authority projection clips,
- future residual learning must be projection-aware.

The distillation matrix method is not ready as a direct controller, but it is diagnostically successful. It showed that low noise and tight `A` are not enough. Distillation matrix adaptation needs Step 2 release protection and Step 3C shadow diagnostics before it should be trusted.

## Why The Methods Are Successful

The project should be judged as a control-design workflow, not only as a single reward number. On that basis, it was successful in both polymer and distillation:

| Case | Successful outcome |
|---|---|
| Polymer | Matrix and structured-matrix RL supervisors improved over MPC after diagnostic caps and release protection. |
| Distillation residual | Residual RL produced physically meaningful improvements under quadratic scoring and exposed the projection/reward mismatch causing late degradation. |
| Distillation matrix | Pre-protection matrix RL revealed that distillation is highly sensitive to `B` authority, motivating release protection and dual-cost diagnostics before transfer. |

The important success is methodological:

1. Baseline MPC gives a stable reference.
2. RL supervisors explore adaptive changes.
3. Offline diagnostics identify dangerous authority directions.
4. Release guards reduce early policy-release damage.
5. Reward sensitivity separates real plant behavior from reward-shaping artifacts.

This is exactly the kind of layered workflow needed for RL in process control. The project did not simply train a black-box controller. It built an interpretable adaptive controller around MPC and used the failures to identify better safety and learning structures.

## Current Recommendations

For polymer:

- Keep Step 1 diagnostics enabled.
- Keep Step 2 release-protected advisory caps enabled.
- Treat Step 3B as informative but not final.
- Move toward Step 3C shadow diagnostics before using hard fallback gates.

For distillation matrix:

- Do not interpret the latest matrix run as a Step 2/3 failure.
- Do not transfer matrix authority directly.
- Run Step 1 and Step 2 before any serious matrix rerun.
- Add Step 3C shadow diagnostics first.

For distillation residual:

- Keep `rho` authority.
- Add projection-aware diagnostics and possibly a projection penalty.
- Report both the current shaped reward and pure quadratic reward.
- Investigate whether the composition-temperature tradeoff should be adjusted in the reward.

## References

1. Rawlings, J. B., Mayne, D. Q., and Diehl, M. M. (2017). *Model Predictive Control: Theory, Computation, and Design*. [PDF](https://sites.engineering.ucsb.edu/~jbraw/mpc/MPC-book-2nd-edition-5th-printing.pdf).
2. Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518, 529-533. [DOI link](https://www.nature.com/articles/nature14236).
3. Lillicrap, T. P., Hunt, J. J., Pritzel, A., et al. (2015). Continuous control with deep reinforcement learning. [arXiv:1509.02971](https://arxiv.org/abs/1509.02971).
4. Fujimoto, S., van Hoof, H., and Meger, D. (2018). Addressing function approximation error in actor-critic methods. *Proceedings of ICML*. [arXiv:1802.09477](https://arxiv.org/abs/1802.09477).
5. Haarnoja, T., Zhou, A., Abbeel, P., and Levine, S. (2018). Soft Actor-Critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. [arXiv:1801.01290](https://arxiv.org/abs/1801.01290).
6. Kober, J., Bagnell, J. A., and Peters, J. (2013). Reinforcement learning in robotics: A survey. *The International Journal of Robotics Research*, 32(11), 1238-1274. [DOI link](https://doi.org/10.1177/0278364913495721).
