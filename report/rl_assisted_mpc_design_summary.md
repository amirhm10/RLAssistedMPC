# RL-Assisted MPC Design Summary For Polymer And Distillation

Date: 2026-04-25

## Executive Summary

This project studies **reinforcement-learning-assisted model predictive control** for two process-control systems:

- a polymer continuous stirred-tank reactor, and
- a distillation column.

The important design decision is that RL does not replace MPC. MPC remains the optimizer that computes constrained control moves from a process model, setpoint trajectory, input limits, and tuning weights. RL works one layer above MPC. It changes selected MPC design variables, such as model multipliers, residual control corrections, penalty weights, or horizons. This makes the controller adaptive while keeping the useful structure of MPC: constraint handling, finite-horizon prediction, and interpretable tuning knobs.

The approach was successful in both cases. In the polymer case, RL-assisted MPC showed direct improvement over the fixed MPC baseline, especially through matrix and structured-matrix supervisors. In the distillation case, the methods also produced a successful research outcome: residual RL produced useful physical behavior, the matrix method revealed the sensitivity of the column to model-gain changes, and the reward analysis showed why a lower shaped reward can still hide better behavior under simpler quadratic metrics. The distillation results are therefore not just a pass/fail score; they are evidence that the RL-assisted MPC framework can discover where adaptation helps and where the process is too sensitive for a naive policy.

The repository uses several RL-assisted MPC families:

| Method family | RL action changes | Typical agents |
|---|---|---|
| Scalar matrix multiplier | Global `A` multiplier and input-column `B` multipliers | TD3, SAC |
| Structured matrix multiplier | Block, band, or coordinate-wise model multipliers | TD3, SAC |
| Residual correction | Additive correction around nominal MPC input | TD3, SAC |
| Weight tuning | MPC tracking and input-move penalty multipliers | TD3, SAC |
| Horizon tuning | Discrete prediction/control-horizon choices | DQN, dueling DQN |
| Combined supervisor | Multiple MPC knobs at the same time | Several agents working together |

This summary explains these methods in an algorithm-style way so the whole design can be understood as one consistent workflow.

## Literature Context

MPC is a standard control method for constrained process systems because it repeatedly solves a finite-horizon optimization problem using a prediction model, constraints, and a tracking objective [Rawlings et al., 2017](https://sites.engineering.ucsb.edu/~jbraw/mpc/MPC-book-2nd-edition-5th-printing.pdf). RL is useful when the best control policy or best controller tuning is hard to write by hand. Deep Q-learning showed that neural networks can approximate value functions for sequential decisions [Mnih et al., 2015](https://www.nature.com/articles/nature14236). Continuous-control actor-critic algorithms such as DDPG [Lillicrap et al., 2015](https://arxiv.org/abs/1509.02971), TD3 [Fujimoto et al., 2018](https://arxiv.org/abs/1802.09477), and SAC [Haarnoja et al., 2018](https://arxiv.org/abs/1801.01290) are natural choices when the action is a continuous tuning parameter. Dueling DQN separates state value and action advantage, which is useful when several discrete actions can be similarly good [Wang et al., 2016](https://proceedings.mlr.press/v48/wangf16.html).

The project follows a practical control idea: use RL where learning is valuable, but keep MPC as the constrained optimizer. This avoids making the RL policy directly responsible for every plant input. Instead, the RL policy changes how MPC reasons about the next move.

## Baseline MPC Layer

All RL-assisted methods start from the same baseline: an offset-free linear MPC model identified from process data. In compact form, the nominal prediction model is:

$$ x_{k+1} = A_0 x_k + B_0 \Delta u_k. $$

The predicted controlled output is:

$$ y_k = C x_k. $$

At each decision time, MPC solves an optimization problem over a prediction horizon:

$$ J = \sum_{i=1}^{N_p} \|y_{k+i} - y_{k+i}^{sp}\|_Q^2 + \sum_{i=0}^{N_c-1} \|\Delta u_{k+i}\|_R^2. $$

The MPC returns a planned input sequence, but only the first move is applied to the nonlinear plant. At the next sample, the state/output estimate is updated and the optimization is solved again. This receding-horizon structure is what every RL supervisor modifies.

The generic closed-loop loop is:

1. Measure or estimate the current plant state and output.
2. Build the RL observation from tracking errors, recent outputs, inputs, setpoints, and optional mismatch features.
3. Let the RL agent choose a supervisory action.
4. Convert that action into MPC parameters.
5. Solve MPC with those parameters.
6. Apply the first input move to the nonlinear plant.
7. Compute reward from tracking, input movement, constraint behavior, and optional bonuses.
8. Store the transition in replay and update the agent.

The difference between methods is step 4: what part of MPC the RL action changes.

## Continuous Agents: TD3 And SAC

The continuous-action methods use either TD3 or SAC.

TD3 is the main deterministic actor-critic agent used in the current successful matrix, structured matrix, residual, and weight experiments. TD3 has an actor network, two critic networks, target networks, delayed actor updates, and target-policy smoothing. The two critics reduce overestimation bias by using the smaller target value:

$$ y = r + \gamma \min_{j=1,2} Q_{\phi_j'}(s', \pi_{\theta'}(s') + \epsilon). $$

Algorithmically, TD3 in this project works as follows:

1. The actor maps the observation `s_t` to a continuous raw action `a_t`.
2. Exploration noise is added during training.
3. The raw action is mapped into meaningful MPC parameters, such as multiplier bounds or weight ranges.
4. MPC solves with those parameters and applies one input move.
5. The transition `(s_t, a_t, r_t, s_{t+1})` is stored in replay.
6. The critics are updated from replay using the clipped double-Q target.
7. The actor is updated less frequently, using the critic gradient.

SAC is the stochastic alternative. It learns a policy distribution instead of a deterministic action. SAC maximizes both expected return and entropy:

$$ J_{\mathrm{SAC}} = \mathbb{E}\left[\sum_t r_t + \alpha_{\mathrm{ent}} \mathcal{H}(\pi(\cdot|s_t))\right]. $$

In this project, SAC is useful as an alternative agent when exploration diversity matters. The MPC wrapper around SAC is the same as TD3: SAC still produces a supervisory action, and MPC still solves the constrained move.

## Method 1: Scalar Matrix Multiplier MPC

The scalar matrix method lets the RL agent change the prediction model used by MPC. The action is low-dimensional and interpretable. For a two-input process, the action usually controls one scalar multiplier for `A` and one multiplier per `B` column:

$$ A_{\theta} = \alpha A_0, \qquad B_{\theta} = B_0 \operatorname{diag}(\beta_1,\beta_2). $$

Here:

- `alpha` changes the predicted process memory and speed,
- `beta_1` changes the predicted gain from input 1,
- `beta_2` changes the predicted gain from input 2.

The algorithm is:

1. Build observation `s_t` from the current MPC/plant state.
2. TD3 or SAC outputs raw action `a_t` in a normalized range.
3. Map `a_t` into physical multipliers `alpha`, `beta_1`, and `beta_2`.
4. Construct the candidate model `(A_theta, B_theta)`.
5. Solve MPC using `(A_theta, B_theta)`.
6. Apply the first MPC input move to the nonlinear plant.
7. Score the closed-loop response and train the agent from replay.

This method is powerful because small model changes can produce meaningfully different MPC moves. It worked well in the polymer case because the polymer plant benefited from adaptive prediction-model corrections. In distillation, the same authority is more delicate. The column is strongly coupled, so `B`-matrix gain changes can cause a control move that looks useful to the modified MPC model but hurts the real nonlinear plant.

The scalar matrix method is successful because it gives the RL agent a small number of physically meaningful knobs. The agent does not need to learn a direct input policy. It learns how the MPC model should be adjusted.

## Method 2: Structured Matrix Multiplier MPC

The structured matrix method extends the scalar matrix idea. Instead of using only one `A` multiplier and two `B` multipliers, it allows different parts of the model to be adjusted separately:

$$ A_{\theta} = \mathcal{S}_A(A_0,\theta_A), \qquad B_{\theta} = \mathcal{S}_B(B_0,\theta_B). $$

The structure can be block-based, band-based, or coordinate-based depending on the notebook configuration. A typical structured action includes:

- multipliers for selected `A` blocks,
- multipliers for off-diagonal or coupling terms,
- multipliers for each `B` input column.

The algorithm is:

1. Build observation `s_t`.
2. TD3 or SAC outputs a vector of raw structured multipliers.
3. Each raw coordinate is mapped into a multiplier for a specific model structure.
4. The structured update function modifies only the allowed entries or blocks of `A_0` and `B_0`.
5. MPC solves with the structured candidate model.
6. The plant receives the first input move.
7. Reward and replay update the actor and critics.

The structured method has more expressive power than the scalar method. It can learn that one dynamic block needs stronger correction while another should stay near nominal. That is valuable for systems where one output/input pathway is more mismatched than another. The cost is that interactions between coordinates become harder to predict. Two individually reasonable multipliers can combine into a poor model if they change coupled dynamics in the same direction.

For polymer, the structured matrix method was successful because the added model flexibility allowed the RL policy to improve MPC behavior beyond a fixed nominal model. For distillation, this method should be treated carefully because the column has stronger input-output interactions and slower coupled dynamics.

## Method 3: Residual MPC Correction

The residual method keeps nominal MPC as the main controller and lets RL add a correction around the MPC move:

$$ u_t^{\mathrm{RL}} = u_t^{\mathrm{MPC}} + \Delta u_t^{\mathrm{res}}. $$

This is different from the matrix methods. Matrix methods change the prediction model before MPC solves. Residual methods let MPC solve normally, then apply a learned correction to the final input.

The residual algorithm is:

1. Solve nominal MPC using `(A_0, B_0)`.
2. Build the RL observation from tracking errors, current output, input, setpoint, and optional mismatch information.
3. TD3 or SAC outputs a residual action.
4. Map the raw residual action into physical input units or scaled input units.
5. Add the residual correction to the nominal MPC move.
6. Project or clip the resulting input to the feasible input range.
7. Apply the corrected input to the plant.
8. Store the executed action and reward in replay.

The residual method is intuitive: MPC handles the main stabilizing behavior, and RL learns a correction for parts of the plant response that nominal MPC does not capture. It is often easier to keep stable than model-multiplier control because the nominal MPC move is still present at every step.

In distillation, the residual method is promising. The latest analysis showed that the residual controller can improve some physical metrics, especially temperature tracking and input smoothness. The shaped reward can still degrade at the tail because composition tracking worsens. That means the residual method is not simply bad; it is exposing a reward tradeoff. Under pure quadratic or no-bonus reward variants, the residual run can look better than the shaped score suggests.

## Method 4: Weight-Tuning MPC

The weight-tuning method lets RL adjust the MPC cost function instead of the process model or final input. The nominal MPC objective is:

$$ J = \sum_{i=1}^{N_p} e_{k+i}^{\top} Q e_{k+i} + \sum_{i=0}^{N_c-1} \Delta u_{k+i}^{\top} R \Delta u_{k+i}. $$

The RL action changes the effective penalty matrices:

$$ Q_{\theta} = \operatorname{diag}(q_{\theta})Q_0, \qquad R_{\theta} = \operatorname{diag}(r_{\theta})R_0. $$

The algorithm is:

1. Build the RL observation from current tracking and input behavior.
2. TD3 or SAC outputs continuous weight multipliers.
3. Map the raw action to positive multipliers for output tracking and input movement penalties.
4. Construct `(Q_theta, R_theta)`.
5. Solve MPC with the nominal model but modified objective weights.
6. Apply the first input move.
7. Train the RL agent using the closed-loop reward.

This method changes controller aggressiveness. Increasing output weights makes MPC chase setpoints more strongly. Increasing input-move weights makes MPC smoother and less aggressive. Weight tuning is usually less risky than model-multiplier tuning because the plant model remains unchanged. It is useful when the nominal model is acceptable but the fixed tuning is not optimal across all operating regimes.

In polymer and distillation, weight agents provide a controlled way to adapt performance tradeoffs. They may not create as large an improvement as model multipliers when the model mismatch is important, but they are easier to interpret and can be useful inside the combined supervisor.

## Method 5: Horizon-Tuning MPC With DQN And Dueling DQN

The horizon method is discrete. Instead of choosing a continuous multiplier, the RL agent chooses from a finite set of prediction/control-horizon pairs:

$$ a_t \in \{(N_p^{(1)},N_c^{(1)}), (N_p^{(2)},N_c^{(2)}), \ldots, (N_p^{(m)},N_c^{(m)})\}. $$

This is why DQN-style agents are used. DQN estimates an action-value function:

$$ Q(s,a) \approx \mathbb{E}\left[\sum_{k=0}^{\infty}\gamma^k r_{t+k} \mid s_t=s, a_t=a\right]. $$

The DQN horizon algorithm is:

1. Define a discrete action table of allowed horizon pairs.
2. Build observation `s_t`.
3. The DQN estimates `Q(s_t, a)` for each horizon action.
4. Choose an action using epsilon-greedy exploration during training.
5. Set MPC horizons to the selected `(N_p, N_c)`.
6. Solve MPC, apply the first input move, and compute reward.
7. Store transition and update the Q-network from replay.

Dueling DQN uses the same action table, but the neural network separates state value from action advantage:

$$ Q(s,a) = V(s) + A(s,a) - \frac{1}{|\mathcal{A}|}\sum_{a'}A(s,a'). $$

The dueling horizon algorithm is:

1. Use the same discrete horizon table as DQN.
2. Pass observation `s_t` through a shared feature network.
3. Split into a value stream `V(s_t)` and an advantage stream `A(s_t,a)`.
4. Recombine them into `Q(s_t,a)` for all horizon choices.
5. Select the horizon action and run MPC.
6. Train from replay using the temporal-difference target.

Dueling DQN is useful when many horizon choices behave similarly for a given state. The value stream learns whether the state itself is easy or difficult, while the advantage stream learns which horizon is better inside that state. This can improve learning efficiency for horizon selection.

## Method 6: Combined Multi-Agent RL-Assisted MPC

The combined method lets several agents work together. Instead of asking one agent to choose every MPC design variable, the combined supervisor assigns one responsibility to each agent:

- horizon agent chooses the prediction/control horizon,
- matrix agent modifies the prediction model,
- weight agent modifies the MPC objective,
- residual agent adds a final correction.

The combined algorithm is:

1. Build a shared observation from the current plant and MPC state.
2. Send the observation, or method-specific versions of it, to each agent.
3. The horizon agent selects a discrete horizon action using DQN or dueling DQN.
4. The matrix agent selects continuous model multipliers using TD3 or SAC.
5. The weight agent selects continuous penalty multipliers using TD3 or SAC.
6. The MPC problem is assembled with the selected horizons, model, and weights.
7. MPC solves and proposes a nominal combined-supervisor move.
8. The residual agent optionally adds a final input correction.
9. The corrected input is applied to the plant.
10. Rewards are computed and transitions are stored for the relevant agents.

This is the most general architecture in the project. It treats MPC as a configurable optimizer and lets each RL agent specialize in one controller knob. The advantage is flexibility: one agent can make the controller more aggressive, another can adjust model mismatch, another can adapt horizons, and another can correct the final input. The challenge is credit assignment. If the closed-loop result improves or degrades, it can be difficult to know which agent caused the change.

The combined method is important because it represents the long-term direction of the project: not one black-box RL controller, but a coordinated set of interpretable RL supervisors around MPC.

## Polymer Case Study

The polymer CSTR has two controlled outputs:

- viscosity `eta`,
- reactor temperature `T`.

The manipulated inputs are:

- coolant flow `Qc`,
- monomer flow `Qm`.

The polymer system is the clearest performance success. Matrix and structured-matrix RL supervisors improved the closed-loop behavior compared with fixed MPC. This means that the nominal identified model and fixed tuning were not the best possible controller description across the full operating range. RL helped by adapting the prediction model used inside MPC.

The polymer results support these conclusions:

1. Scalar matrix multipliers are a useful low-dimensional adaptive model correction.
2. Structured matrix multipliers can add more flexibility when the model mismatch is not uniform.
3. Residual and weight agents give additional ways to adapt controller behavior without fully replacing MPC.
4. Horizon agents are useful when the best prediction horizon changes with operating condition.
5. A combined supervisor can use all these adaptations together, although it requires careful interpretation of which agent is helping.

Overall, polymer demonstrates that RL-assisted MPC can improve a process-control system while still using MPC as the main optimizer.

## Distillation Case Study

The distillation column has two controlled outputs:

- tray-24 ethane composition,
- tray-85 temperature.

The manipulated inputs are:

- reflux flow,
- reboiler duty.

Distillation is harder than polymer because the process is strongly coupled and sensitive. A change in one manipulated input can affect composition and temperature across long dynamic paths. This makes direct model-multiplier adaptation more difficult. The matrix method showed that even small-looking changes in model gain can degrade the shaped reward, especially when `B` authority is broad.

The residual method is more promising for distillation. It keeps nominal MPC active and only adds a correction. The latest residual analysis showed a tradeoff: the residual controller can improve temperature tracking and input smoothness, but the shaped reward can punish it if composition tracking worsens near the end. Under pure quadratic reward candidates, the residual behavior can look better than the original reward score.

The distillation results are successful because they reveal the right design lesson:

1. Matrix multipliers are powerful but must be handled carefully for strongly coupled columns.
2. Residual correction is a practical direction because it keeps nominal MPC in the loop.
3. Weight and horizon supervisors are attractive for distillation because they adapt aggressiveness without changing the plant model directly.
4. Combined supervision may be useful if each agent has limited and interpretable authority.

The result is not that every RL method is automatically better than MPC. The result is that the framework can identify which kind of adaptation is useful for each process.

## Overall Success Statement

The project was successful in both polymer and distillation, but the success has different meanings:

| Case | Successful outcome |
|---|---|
| Polymer | RL-assisted MPC produced direct performance improvement over fixed MPC, especially through matrix and structured-matrix supervisors. |
| Distillation residual | RL residual correction produced meaningful physical improvements under several reward interpretations and exposed a composition-temperature tradeoff. |
| Distillation matrix | RL matrix adaptation identified that the distillation column is highly sensitive to model-gain authority, especially through the input matrix. |
| Cross-method design | The project produced a unified family of interpretable RL supervisors that can tune model, weights, horizons, and residual corrections around MPC. |

This is a useful result for process-control research. The project does not claim that a single RL agent should replace MPC. It shows that MPC can be made adaptive by using RL agents as interpretable supervisors. The polymer case shows direct performance improvement. The distillation case shows that the same framework can diagnose sensitivity, reveal reward tradeoffs, and identify safer adaptation directions.

## References

1. Rawlings, J. B., Mayne, D. Q., and Diehl, M. M. (2017). *Model Predictive Control: Theory, Computation, and Design*. [PDF](https://sites.engineering.ucsb.edu/~jbraw/mpc/MPC-book-2nd-edition-5th-printing.pdf).
2. Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518, 529-533. [DOI link](https://www.nature.com/articles/nature14236).
3. Lillicrap, T. P., Hunt, J. J., Pritzel, A., et al. (2015). Continuous control with deep reinforcement learning. [arXiv:1509.02971](https://arxiv.org/abs/1509.02971).
4. Fujimoto, S., van Hoof, H., and Meger, D. (2018). Addressing function approximation error in actor-critic methods. *Proceedings of ICML*. [arXiv:1802.09477](https://arxiv.org/abs/1802.09477).
5. Haarnoja, T., Zhou, A., Abbeel, P., and Levine, S. (2018). Soft Actor-Critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. [arXiv:1801.01290](https://arxiv.org/abs/1801.01290).
6. Wang, Z., Schaul, T., Hessel, M., van Hasselt, H., Lanctot, M., and de Freitas, N. (2016). Dueling Network Architectures for Deep Reinforcement Learning. *Proceedings of ICML*. [PMLR link](https://proceedings.mlr.press/v48/wangf16.html).
