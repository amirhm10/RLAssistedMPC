# Markov Dynamic-Matrix MPC And Supervisor-Gated TD3: Detailed Method Report

Date: 2026-06-02

Case studies: polymer CSTR and Aspen Dynamics C2 splitter distillation column

Scope: detailed method explanation, not a new result claim

## Objective

This report explains the Markov method from the beginning, assuming the reader does not already know dynamic-matrix MPC. It also writes the mathematics of the supervisor-gated TD3 algorithm used in the current polymer and distillation Markov critic-warm runners.

The central idea is simple:

1. MPC still computes the physical input move.
2. The Markov method changes the finite-horizon prediction model that MPC uses.
3. The vector \(z\) is a small, bounded correction to that prediction model.
4. TD3 can propose \(z\), but SG-TD3 makes the actor compete with a supervisor \(z\) before execution.
5. The supervisor is LS-or-MPC in the current Markov SG-TD3 runners: use the online least-squares Markov correction when it is accepted, otherwise use the nominal MPC correction \(z=0\).

## Files Inspected

Reports inspected:

- `report/06_supervisor_gated_td3.md`
- `report/polymer_markov_unified_algorithm_and_tuning_start_2026_05_11.md`
- `report/distillation_markov_td3_decision_authority_2026_05_17.md`
- `report/distillation_polymer_markov_sg_td3_outcome_2026_06_02.md`

Implementation files inspected:

- `utils/markov_runner.py`
- `utils/supervisor_gated_action.py`
- `TD3Agent/supervisor_gated_agent.py`
- `distillation_RL_assisted_MPC_markov_unified.py`
- `distillation_RL_assisted_MPC_markov_supervisor_gated_td3_critic_warm_unified.py`
- `RL_assisted_MPC_markov_supervisor_gated_td3_critic_warm_unified.py`
- `systems/distillation/notebook_params.py`

## Big Picture

The project uses RL-assisted MPC, not direct black-box RL control.

MPC is the inner controller. It predicts future outputs, optimizes a sequence of input moves, respects constraints, and applies only the first move. RL is the outer supervisor. It does not normally send coolant flow, monomer flow, reflux, or reboiler duty directly to the plant. Instead, RL changes a controller quantity such as:

- horizons,
- weights,
- residual input correction,
- model multipliers,
- Markov dynamic-matrix correction.

The Markov method belongs to the last category. It changes the matrix that maps future input moves to future output predictions.

## 1. Basic Plant And MPC Notation

At each sample \(k\), the controller sees or estimates the current process condition. The controlled output is

$$ y_k \in \mathbb{R}^{n_y}. $$

The manipulated input is

$$ u_k \in \mathbb{R}^{n_u}. $$

The input move is

$$ \Delta u_k = u_k-u_{k-1}. $$

The setpoint is

$$ y_k^{\mathrm{sp}} \in \mathbb{R}^{n_y}. $$

For the polymer CSTR:

- \(y_k=[\eta_k,T_k]^\top\), where \(\eta\) is viscosity and \(T\) is reactor temperature.
- \(u_k=[Q_{c,k},Q_{m,k}]^\top\), where \(Q_c\) is coolant flow and \(Q_m\) is monomer flow.

For the distillation column:

- \(y_k=[x_{24,k},T_{85,k}]^\top\), where \(x_{24}\) is tray-24 ethane composition and \(T_{85}\) is tray-85 temperature.
- \(u_k=[L_k,V_k]^\top\), where \(L\) is reflux flow and \(V\) is reboiler duty or boilup-related manipulated input in the Aspen adapter.

The controller uses scaled deviation coordinates. That means the controller usually does not work directly with physical \(u_k\) and \(y_k\). It works with variables measured relative to a steady state and scaled by stored min-max or system-identification artifacts.

Conceptually:

$$ \Delta y_k^{\mathrm{scaled}} = \mathrm{scale}(y_k)-\mathrm{scale}(y_{\mathrm{ss}}). $$

$$ \Delta u_k^{\mathrm{scaled}} = \mathrm{scale}(u_k)-\mathrm{scale}(u_{\mathrm{ss}}). $$

This matters because an RL action such as \(z=0.05\) is not a physical tray temperature change. It is a dimensionless correction in the scaled model-prediction space.

## 2. Offset-Free Linear Model Used By MPC

The nonlinear plant is controlled by a linear offset-free MPC model. The model has an augmented state:

$$ x_{a,k} \in \mathbb{R}^{n_x}. $$

The nominal augmented model is

$$ x_{a,k+1}=A_{\mathrm{aug}}x_{a,k}+B_{\mathrm{aug}}\Delta u_k. $$

The output equation is

$$ y_k=C_{\mathrm{aug}}x_{a,k}. $$

The observer estimates the augmented state. In compact form:

$$ \hat{x}_{a,k+1}=A_{\mathrm{aug}}\hat{x}_{a,k}+B_{\mathrm{aug}}\Delta u_k+L(y_k-\hat{y}_k). $$

The innovation is the measurement-prediction mismatch:

$$ \nu_k = y_k-\hat{y}_k. $$

The offset-free augmentation is important because chemical process models often have bias. Without offset correction, MPC may settle at the wrong output even if the linear model is stable.

## 3. What MPC Solves

MPC predicts \(N_p\) future output samples and optimizes \(N_c\) future input moves.

The stacked future output vector is

$$ Y_k=\begin{bmatrix}y_{k+1}^\top & y_{k+2}^\top & \cdots & y_{k+N_p}^\top\end{bmatrix}^\top. $$

The stacked future move vector is

$$ \Delta U_k=\begin{bmatrix}\Delta u_k^\top & \Delta u_{k+1}^\top & \cdots & \Delta u_{k+N_c-1}^\top\end{bmatrix}^\top. $$

MPC solves a finite-horizon tracking problem:

$$ \Delta U_k^\star=\arg\min_{\Delta U_k} \sum_{i=1}^{N_p} e_{k+i}^\top Qe_{k+i}+\sum_{i=0}^{N_c-1}\Delta u_{k+i}^\top R\Delta u_{k+i}. $$

where

$$ e_{k+i}=y_{k+i}-y_{k+i}^{\mathrm{sp}}. $$

The constraints are the physical or scaled input bounds:

$$ u_{\min}\le u_{k+i}\le u_{\max}. $$

Only the first move is applied:

$$ u_k = u_{k-1}+\Delta u_k^\star. $$

Then the plant advances, a new measurement arrives, and MPC solves again.

## 4. What Is A Markov Block?

The word Markov here does not mean a Markov decision process. In this controller, a Markov block is an impulse-response or step-response coefficient of the linear prediction model.

Start with the state-space model:

$$ x_{k+1}=Ax_k+B\Delta u_k. $$

$$ y_k=Cx_k. $$

Suppose the current estimated state is zero and we apply one input move \(\Delta u_k\). The output one sample later is

$$ y_{k+1}=CB\Delta u_k. $$

The output two samples later, if no further new moves are applied, is

$$ y_{k+2}=CAB\Delta u_k. $$

The output three samples later is

$$ y_{k+3}=CA^2B\Delta u_k. $$

So the response block \(i\) samples after an input move is

$$ M_i=CA^{i-1}B. $$

In the code this is computed for the augmented offset-free matrices:

$$ M_i=C_{\mathrm{aug}}A_{\mathrm{aug}}^{i-1}B_{\mathrm{aug}}. $$

Each \(M_i\) is a matrix:

$$ M_i\in\mathbb{R}^{n_y\times n_u}. $$

For a two-output, two-input process:

$$ M_i=\begin{bmatrix}m_{i,11} & m_{i,12}\\m_{i,21} & m_{i,22}\end{bmatrix}. $$

The entry \(m_{i,12}\) means:

The predicted effect, \(i\) samples later, of input 2's current move on output 1.

That is the most important intuition. \(M_i\) is not a neural-network weight. It is a finite-horizon process-response coefficient.

## 5. From Markov Blocks To The Dynamic Matrix

Dynamic-matrix control stacks the Markov blocks into one large prediction matrix \(G\). This matrix maps the whole future input-move sequence \(\Delta U_k\) to the whole future output sequence \(Y_k\).

The lifted prediction equation is

$$ Y_k=Y_k^{\mathrm{free}}+G_0\Delta U_k. $$

The term \(Y_k^{\mathrm{free}}\) is what the model predicts from the current state if there are no new planned input moves. The term \(G_0\Delta U_k\) is the part caused by the future planned moves.

For a simple horizon, the dynamic matrix has this block-Toeplitz form:

$$ G_0=\begin{bmatrix}M_1&0&0&\cdots\\M_2&M_1&0&\cdots\\M_3&M_2&M_1&\cdots\\\vdots&\vdots&\vdots&\ddots\\M_{N_p}&M_{N_p-1}&M_{N_p-2}&\cdots\end{bmatrix}. $$

Read this row by row:

- The first predicted output \(y_{k+1}\) depends on the first move \(\Delta u_k\) through \(M_1\).
- The second predicted output \(y_{k+2}\) depends on \(\Delta u_k\) through \(M_2\) and on \(\Delta u_{k+1}\) through \(M_1\).
- The third predicted output \(y_{k+3}\) depends on \(\Delta u_k\) through \(M_3\), on \(\Delta u_{k+1}\) through \(M_2\), and on \(\Delta u_{k+2}\) through \(M_1\).

This is why the Markov method is a dynamic-matrix method. It directly changes the finite-horizon matrix that MPC uses to predict future outputs.

## 6. Why Correct The Dynamic Matrix?

The nominal linear model may not match the nonlinear plant perfectly. The mismatch can come from:

- nonlinear process behavior,
- disturbances,
- imperfect system identification,
- changing operating point,
- Aspen column sensitivity and coupling,
- polymer CSTR gain changes under disturbance.

If \(G_0\) is wrong, MPC optimizes the wrong future. It may choose a move that looks good under the model but is not best for the real plant.

The Markov method tries to correct the prediction matrix:

$$ G_0 \rightarrow G(z_k). $$

Here \(z_k\) is a low-dimensional correction vector. The controller does not let RL freely change every entry of \(G\). That would be too many degrees of freedom and would be unsafe. Instead, it defines a small basis of allowed corrections.

## 7. What Is \(z\)?

The Markov blocks are corrected as

$$ M_i(z_k)=M_{i,0}+\sum_{j=1}^{r}z_{j,k}B_i^{(j)}. $$

Here:

- \(M_{i,0}\) is the nominal Markov block at delay \(i\).
- \(B_i^{(j)}\) is basis correction \(j\) at delay \(i\).
- \(z_{j,k}\) is the coefficient multiplying that basis correction at time \(k\).
- \(r\) is the number of allowed correction coordinates.

The corrected dynamic matrix is then

$$ G(z_k)=\mathrm{Toeplitz}(M_1(z_k),M_2(z_k),\ldots,M_{N_p}(z_k)). $$

The correction vector is bounded:

$$ z_k\in[-z_{\max},z_{\max}]^r. $$

In the current Markov runners, common values are near:

- polymer Markov: \(z_{\max}=0.05\),
- distillation Markov SG-TD3 wrapper: \(z_{\max}=0.05\).

The active basis family in the Markov path is typically `io_pair_gain`. For two outputs and two inputs, that gives four coordinates:

$$ z_k=\begin{bmatrix}z_{y_1u_1,k}&z_{y_1u_2,k}&z_{y_2u_1,k}&z_{y_2u_2,k}\end{bmatrix}^\top. $$

Each coordinate scales one output-input Markov channel across the horizon.

For example, \(z_{y_2u_1,k}>0\) strengthens the model's predicted effect of input 1 on output 2. If this channel was underpredicted by the nominal linear model, a positive correction may help MPC make a better move.

## 8. RL Action Mapping For Markov

The TD3 actor outputs a normalized action:

$$ a_{\theta,k}\in[-1,1]^r. $$

The Markov runner maps it to a dynamic-matrix correction:

$$ z_{\theta,k}=z_{\max}\,\mathrm{clip}(a_{\theta,k},-1,1). $$

The inverse map, used when an LS supervisor correction must be passed to SG-TD3 in normalized action coordinates, is

$$ a_{\mathrm{sup},k}=\mathrm{clip}\left(\frac{z_{\mathrm{sup},k}}{z_{\max}},-1,1\right). $$

This is why actor and supervisor actions can be compared by SG-TD3: both live in the same normalized TD3 action space.

## 9. What Happens After A Candidate \(z\) Is Chosen?

After any candidate \(z\) is chosen, the controller builds \(G(z)\) and solves MPC again:

$$ \Delta U_k^\star(z)=\arg\min_{\Delta U_k} J_k(\Delta U_k;G(z)). $$

The plant still receives an MPC move:

$$ u_k=u_{k-1}+\Delta u_{k}^{\star}(z). $$

So TD3 is not directly selecting reflux, reboiler duty, coolant flow, or monomer flow. TD3 selects \(z\). The MPC optimizer converts \(z\) into a constrained physical input move.

That is the safety logic of the Markov family:

RL changes the prediction model, but MPC remains the final optimizer.

## 10. Online LS Markov Correction

The online least-squares correction is a non-neural supervisor. It asks:

Which \(z\) would have made the model's recent predictions closer to the measured plant outputs?

For each recent time \(\tau\), the runner stores:

- the measured future outputs,
- the previous estimated state,
- the actual input moves that were executed,
- the nominal and corrected predicted output sequences.

The measured stacked output is

$$ Y_\tau^{\mathrm{meas}}=\begin{bmatrix}y_{\tau+1}^\top&\cdots&y_{\tau+N_p}^\top\end{bmatrix}^\top. $$

The actual move sequence from that time is

$$ \Delta U_\tau^{\mathrm{real}}=\begin{bmatrix}\Delta u_\tau^\top&\cdots&\Delta u_{\tau+N_c-1}^\top\end{bmatrix}^\top. $$

The corrected prediction is

$$ Y_\tau^{\mathrm{pred}}(z)=Y_\tau^{\mathrm{free}}+G(z)\Delta U_\tau^{\mathrm{real}}. $$

The LS objective is

$$ z_k^{\mathrm{LS}}=\arg\min_z \sum_{\tau\in\mathcal{W}_k}\left\lVert W_y\left(Y_\tau^{\mathrm{meas}}-Y_\tau^{\mathrm{pred}}(z)\right)\right\rVert_2^2+\lambda_z\left\lVert z\right\rVert_2^2. $$

The regularization term \(\lambda_z\left\lVert z\right\rVert_2^2\) discourages large model corrections unless they clearly improve prediction.

The prediction-improvement score compares nominal and corrected prediction errors:

$$ S_{\mathrm{pred}}(z)=\sum_{\tau\in\mathcal{W}_k}\left(\left\lVert W_yE_0(\tau)\right\rVert_2^2-\left\lVert W_yE_z(\tau)\right\rVert_2^2\right)-\lambda_z\left\lVert z\right\rVert_2^2. $$

where

$$ E_0(\tau)=Y_\tau^{\mathrm{meas}}-\left(Y_\tau^{\mathrm{free}}+G_0\Delta U_\tau^{\mathrm{real}}\right). $$

and

$$ E_z(\tau)=Y_\tau^{\mathrm{meas}}-\left(Y_\tau^{\mathrm{free}}+G(z)\Delta U_\tau^{\mathrm{real}}\right). $$

If \(S_{\mathrm{pred}}(z)>0\), the corrected model predicts the recent data better than the nominal model after accounting for regularization.

Older guarded Markov variants used this score as a hard release condition. The latest SG-TD3 Markov design treats LS as a supervisor candidate, not as a manually imposed live safety layer.

## 11. Markov Candidate Types

At a given time step, the Markov runner can have several candidate corrections:

Nominal correction:

$$ z_{\mathrm{mpc},k}=0. $$

This means no Markov correction and therefore \(G(z)=G_0\).

Least-squares correction:

$$ z_{\mathrm{LS},k}=\arg\min_z \mathrm{recent\ prediction\ error}. $$

This means a local data-driven correction fitted from recent history.

TD3 actor correction:

$$ z_{\theta,k}=z_{\max}\mathrm{clip}(\mu_\theta(s_k),-1,1). $$

This means a learned correction chosen to maximize long-run reward.

The scientific tension is that \(z_{\mathrm{LS}}\) is usually good at short-window prediction, while \(z_{\theta}\) may eventually be better for the episode reward. SG-TD3 is a way to decide between them online.

## 12. RL State Used By Markov TD3

The Markov TD3 state is a feature vector built from the current closed-loop condition. The implementation includes:

- current model or observer state,
- tracking error,
- innovation,
- previous input deviation,
- previous executed Markov correction \(z_{k-1}\),
- current LS correction \(z_{\mathrm{LS},k}\),
- LS prediction score,
- LS gain-drift diagnostic.

In compact notation:

$$ s_k^{\mathrm{M}}=\phi(\hat{x}_{a,k},e_k,\nu_k,u_{k-1},z_{k-1},z_{\mathrm{LS},k},S_{\mathrm{LS},k},d_{\mathrm{gain},k}). $$

The exact normalization is handled by the runner's state conditioner and mismatch feature settings.

## 13. Standard TD3 Refresher

TD3 has:

- one deterministic actor \(\mu_\theta(s)\),
- two critics \(Q_{\phi_1}(s,a)\) and \(Q_{\phi_2}(s,a)\),
- target networks \(\mu_{\bar{\theta}}\), \(Q_{\bar{\phi}_1}\), and \(Q_{\bar{\phi}_2}\).

The actor proposes

$$ a_k=\mu_\theta(s_k)+\epsilon_k. $$

The transition stored in replay is

$$ (s_k,a_k,r_k,s_{k+1},d_k). $$

TD3 uses target policy smoothing:

$$ \tilde{a}_{k+1}=\mathrm{clip}(\mu_{\bar{\theta}}(s_{k+1})+\epsilon,-a_{\max},a_{\max}). $$

The critic target is

$$ y_k=r_k+\gamma(1-d_k)\min_{i=1,2}Q_{\bar{\phi}_i}(s_{k+1},\tilde{a}_{k+1}). $$

Each critic minimizes

$$ \mathcal{L}_{Q_i}=\mathbb{E}_{\mathcal{D}}\left[\left(Q_{\phi_i}(s_k,a_k)-y_k\right)^2\right]. $$

The actor is updated less often than the critics. The basic actor loss is

$$ \mathcal{L}_{\pi}=-\mathbb{E}_{\mathcal{D}}\left[Q_{\phi_1}(s_k,\mu_\theta(s_k))\right]. $$

TD3 helps reduce overestimated Q-values by using the smaller of two target critics.

## 14. Why Standard TD3 Was Not Enough

Standard TD3 has a simple live-execution rule:

Execute the actor action when the policy is released.

For Markov, that means:

$$ a_k^{\mathrm{exec}}=a_{\theta,k}. $$

and therefore

$$ z_k^{\mathrm{exec}}=z_{\max}a_{\theta,k}. $$

This can work late in training when the actor has learned a useful correction. But early after warm start, the actor and critics may still be weak. The distillation no-safeguard Markov run showed exactly this problem: full TD3 release had strong late upside but also severe post-warm failure episodes.

The lesson is not that TD3 is useless. The lesson is that actor-only execution is too abrupt for a sensitive process such as the distillation column.

## 15. Supervisor-Gated TD3: Main Idea

SG-TD3 adds a second candidate action:

$$ a_{\mathrm{rl},k}=\mu_\theta(s_k). $$

$$ a_{\mathrm{sup},k}=\mu_{\mathrm{sup}}(s_k). $$

Both are normalized action vectors in the same action space:

$$ a_{\mathrm{rl},k},a_{\mathrm{sup},k}\in[-a_{\max},a_{\max}]^{n_a}. $$

The actor action is the learned TD3 proposal. The supervisor action is a meaningful fallback action.

For different RL-assisted MPC families:

- Weights: the supervisor is the identity multiplier action, which recovers fixed MPC tuning.
- Residual: the supervisor is zero residual, which recovers nominal MPC.
- Markov: the supervisor is LS-or-MPC \(z\), which means LS correction if accepted and nominal \(z=0\) otherwise.

The Markov supervisor in the current critic-warm wrappers is:

$$ z_{\mathrm{sup},k}=\begin{cases}z_{\mathrm{LS},k},&\mathrm{if\ LS\ candidate\ is\ accepted},\\0,&\mathrm{otherwise}.\end{cases} $$

Then the normalized supervisor action is

$$ a_{\mathrm{sup},k}=\mathrm{clip}\left(\frac{z_{\mathrm{sup},k}}{z_{\max}},-1,1\right). $$

## 16. SG-TD3 Critic Score

Both candidate actions are evaluated by the same two TD3 critics.

For a candidate \(a\), the two critic values are:

$$ Q_1(s_k,a),\qquad Q_2(s_k,a). $$

The conservative value is

$$ Q_{\min}(s_k,a)=\min(Q_1(s_k,a),Q_2(s_k,a)). $$

The critic disagreement is

$$ D_Q(s_k,a)=\lvert Q_1(s_k,a)-Q_2(s_k,a)\rvert. $$

The SG-TD3 score is

$$ S(s_k,a)=Q_{\min}(s_k,a)-\rho_QD_Q(s_k,a)-\kappa_{\mathrm{prev}}\left\lVert a-a_{\mathrm{prev},k}\right\rVert_2^2-\kappa_{\mathrm{sup}}\left\lVert a-a_{\mathrm{sup},k}\right\rVert_2^2. $$

Each term has a control meaning:

- \(Q_{\min}\): prefer high predicted return, using the conservative lower critic.
- \(\rho_QD_Q\): penalize actions where the critics disagree.
- \(\kappa_{\mathrm{prev}}\left\lVert a-a_{\mathrm{prev}}\right\rVert_2^2\): penalize sudden action jumps.
- \(\kappa_{\mathrm{sup}}\left\lVert a-a_{\mathrm{sup}}\right\rVert_2^2\): penalize moving far away from the supervisor.

The current polymer and distillation Markov critic-warm wrappers use:

$$ \rho_Q=0.5,\qquad \kappa_{\mathrm{sup}}=0.02,\qquad \kappa_{\mathrm{prev}}=0.01,\qquad \epsilon_A=0. $$

## 17. SG-TD3 Selection Rule

The actor advantage over the supervisor is

$$ A_{\mathrm{rl}\mid\mathrm{sup}}(s_k)=S(s_k,a_{\mathrm{rl},k})-S(s_k,a_{\mathrm{sup},k}). $$

The policy action is selected only if

$$ S(s_k,a_{\mathrm{rl},k})>S(s_k,a_{\mathrm{sup},k})+\epsilon_A. $$

Equivalently:

$$ A_{\mathrm{rl}\mid\mathrm{sup}}(s_k)>\epsilon_A. $$

If this condition is false, the supervisor is selected:

$$ a_k^{\mathrm{exec}}=a_{\mathrm{sup},k}. $$

The current config has `default_to_supervisor = True`, so ties go to the supervisor. This is important near steady state because many small actions can have nearly equal value. If the actor cannot clearly beat the supervisor, the controller stays with the safer action.

## 18. SG-TD3 Replay

SG-TD3 stores the executed action as the transition action:

$$ \mathcal{D}\leftarrow(s_k,a_k^{\mathrm{exec}},r_k,s_{k+1},d_k). $$

It also stores metadata:

$$ a_{\mathrm{rl},k},\quad a_{\mathrm{sup},k},\quad a_{\mathrm{prev},k},\quad S_{\mathrm{rl},k},\quad S_{\mathrm{sup},k},\quad A_{\mathrm{rl}\mid\mathrm{sup},k},\quad \mathrm{source}_k. $$

This is not just software bookkeeping. It is scientifically important.

If the supervisor action is executed, the next plant state was caused by the supervisor action. Training the critic as if the actor action caused that transition would create a false counterfactual sample. The current implementation avoids that mistake.

## 19. SG-TD3 Critic Update

The critic target remains standard one-step TD3:

$$ y_k=r_k+\gamma(1-d_k)\min_iQ_{\bar{\phi}_i}(s_{k+1},\tilde{a}_{k+1}). $$

But the critic is evaluated on the executed action:

$$ \mathcal{L}_{Q_i}=\mathbb{E}_{\mathcal{D}}\left[\left(Q_{\phi_i}(s_k,a_k^{\mathrm{exec}})-y_k\right)^2\right]. $$

The supervisor action is not used as a target action in this SG-TD3 version. It is used for execution selection and optional actor regularization.

## 20. Optional SG-TD3 Actor Regularization

The implementation can add a supervisor-aware actor term:

$$ \mathcal{L}_{\pi}=-\mathbb{E}\left[Q_{\mathrm{mode}}(s,\mu_\theta(s))\right]+\lambda_{\mathrm{sup}}\mathbb{E}\left[w_{\mathrm{sup}}(s)\left\lVert\mu_\theta(s)-a_{\mathrm{sup}}\right\rVert_2^2\right]+\lambda_{\Delta a}\mathbb{E}\left[\left\lVert\mu_\theta(s)-a_{\mathrm{prev}}\right\rVert_2^2\right]. $$

The supervisor imitation weight is

$$ w_{\mathrm{sup}}(s)=\sigma\left(\frac{\epsilon_A-A_{\mathrm{rl}\mid\mathrm{sup}}(s)}{\tau_{\mathrm{sup}}}\right). $$

This weight is large when the actor does not beat the supervisor. It is small when the actor clearly beats the supervisor.

In the current Markov critic-warm wrappers:

$$ \lambda_{\mathrm{sup}}=0. $$

That means the supervisor actor loss is disabled. The safety effect comes from execution gating, not from forcing the actor to imitate the supervisor during the actor update.

## 21. Warm Start And Critic Warm

The current Markov SG-TD3 critic-warm runners use:

- 10 warm-start episodes,
- 3 post-warm-start action-freeze subepisodes,
- 3 post-warm-start actor-freeze subepisodes.

During warm start, the actor is not allowed to control the plant. The controller executes the supervisor.

During critic warm, the actor can be evaluated and replay can grow, but executed actions are still forced to the supervisor. This lets the critics train before the actor is trusted.

In symbols, let \(k_{\mathrm{warm}}\) be the final warm-start step and let \(H_{\mathrm{cw}}\) be the critic-warm duration. For

$$ k\le k_{\mathrm{warm}}+H_{\mathrm{cw}}, $$

the executed SG action is forced:

$$ a_k^{\mathrm{exec}}=a_{\mathrm{sup},k}. $$

After that, the gate is active:

$$ a_k^{\mathrm{exec}}=\begin{cases}a_{\mathrm{rl},k},&S(s_k,a_{\mathrm{rl},k})>S(s_k,a_{\mathrm{sup},k})+\epsilon_A,\\a_{\mathrm{sup},k},&\mathrm{otherwise}.\end{cases} $$

The current distillation wrapper also sets `force_td3_respects_warm_start = True`, so even if TD3 forcing is enabled elsewhere, warm start remains protected.

## 22. Current Distillation Markov SG-TD3 Runner

The current distillation runner is:

`distillation_RL_assisted_MPC_markov_supervisor_gated_td3_critic_warm_unified.py`

It sets:

- `agent_kind = "sg_td3"`
- `run_mode = "disturb"`
- `disturbance_profile = "fluctuation"`
- `warm_start_override = 10`
- `post_warm_start_action_freeze_subepisodes = 3`
- `post_warm_start_actor_freeze_subepisodes = 3`
- `markov_supervisor_mode = "ls_else_mpc"`
- `markov_live_safety_mode = "shadow_only"`
- `z_bound = 0.05`

It disables the manually created live safety layers:

- behavioral cloning,
- BC release gate,
- BC handoff,
- tail anchor,
- live \(z\)-safety projection,
- TD3 priority fallback,
- TD3 authority ramp.

It keeps shadow diagnostics:

- original \(z\)-safety config is saved into shadow safety,
- original TD3 priority fallback config is saved into shadow safety,
- original BC handoff config is saved into shadow safety,
- shadow diagnostics do not alter the executed action.

This is important experimentally. The run isolates the SG-TD3 gate. It does not mix the gate with the earlier hand-built live safety layers.

## 23. Why Shadow Diagnostics Matter

Shadow diagnostics answer:

What would the old safety layer have done, without actually changing the executed action?

For example, the runner can log whether a \(z\)-safety projection would have clipped a requested Markov correction. But if live safety is disabled, the projection does not change:

$$ z_k^{\mathrm{exec}}. $$

This lets us later audit whether the actor was often requesting actions that old safeguards considered risky, while keeping the experimental intervention clean.

## 24. Full Markov SG-TD3 Algorithm

At each step:

1. Measure the plant output \(y_k\).
2. Update the offset-free observer \(\hat{x}_{a,k}\).
3. Solve nominal MPC using \(G_0\).
4. Fit or update the LS Markov candidate \(z_{\mathrm{LS},k}\), if enough history exists.
5. Build the Markov RL state \(s_k^{\mathrm{M}}\).
6. Build the supervisor correction \(z_{\mathrm{sup},k}\): LS if accepted, otherwise zero.
7. Convert \(z_{\mathrm{sup},k}\) to normalized \(a_{\mathrm{sup},k}\).
8. Ask the TD3 actor for \(a_{\mathrm{rl},k}\).
9. During warm start or critic warm, force \(a_k^{\mathrm{exec}}=a_{\mathrm{sup},k}\).
10. Otherwise compute SG-TD3 scores and choose actor or supervisor.
11. Convert \(a_k^{\mathrm{exec}}\) to \(z_k^{\mathrm{exec}}\).
12. Build \(G(z_k^{\mathrm{exec}})\).
13. Solve MPC with \(G(z_k^{\mathrm{exec}})\).
14. Apply the first MPC move to the nonlinear plant.
15. Compute reward.
16. Store the executed transition and SG metadata.
17. Train the critics, and update the actor only after the configured delay and freeze period.

## 25. Why This Helps Safety

SG-TD3 helps safety in an operational sense, not a theorem-proving sense.

It helps because:

- the actor is not automatically executed after warm start,
- the supervisor wins if critic evidence is weak,
- critic disagreement is penalized,
- action jumps away from the previous action are penalized,
- action jumps away from the supervisor are penalized,
- replay stores executed actions correctly,
- action-source logs show whether RL is actually controlling or the supervisor is carrying the run.

For Markov, the safety benefit is especially meaningful because the actor is changing a model used by MPC. A bad \(z\) can cause MPC to optimize against a distorted prediction matrix. SG-TD3 makes the actor prove that its \(z\) is better than the LS-or-MPC supervisor under the learned critics.

## 26. What SG-TD3 Does Not Guarantee

SG-TD3 does not guarantee closed-loop stability.

It does not guarantee that the critic is correct.

It does not guarantee that the LS supervisor is always safe.

It does not guarantee that a positive score means good tracking.

It reduces one specific risk: actor-only execution when the actor has not earned authority.

The method still needs closed-loop validation using:

- reward curves,
- tracking metrics,
- action-source fractions,
- \(z\)-norm logs,
- prediction score,
- gain drift,
- shadow safety diagnostics,
- final-episode overlays,
- post-warm and tail-window summaries.

## 27. How To Read A Markov SG-TD3 Result Bundle

The most important logs are:

- `agent_kind`: should be `sg_td3`.
- `markov_supervisor_mode`: should be `ls_else_mpc`.
- `markov_live_safety_mode`: should be `shadow_only` for the clean SG-TD3 ablation.
- `z_safety.enabled`: should be false for the no-live-manual-safety SG-TD3 wrapper.
- `td3_priority_fallback.enabled`: should be false for the same wrapper.
- `markov_shadow_safety_enabled`: should be true.
- `sg_selected_source_log`: tells whether actor or supervisor was selected.
- `sg_supervisor_kind_log`: tells whether the supervisor was LS or MPC.
- `z_executed_log`: tells the Markov correction that actually changed the MPC prediction.
- `shadow_z_safety_*` logs: tell what the old safety projection would have done.
- reward and output logs: tell whether the closed-loop behavior improved.

The key scientific question is not just:

Did reward improve?

It is:

Who had authority when reward improved, and what kind of \(z\) was being executed?

## 28. Main Interpretation

The Markov method is a dynamic-matrix adaptation method. It changes \(G\), the lifted finite-horizon matrix that predicts how planned input moves affect future outputs.

The vector \(z\) is the low-dimensional control knob for that adaptation. It does not directly move the plant. It changes the prediction matrix, then MPC solves the constrained input problem.

Least-squares \(z_{\mathrm{LS}}\) is a local prediction-error correction. TD3 \(z_\theta\) is a learned long-run-reward correction. SG-TD3 is the execution mechanism that decides whether the learned correction deserves to beat the supervisor correction.

For polymer, the current SG-TD3 Markov result shows the desired pattern: supervisor dominance early and more actor authority later. For distillation, the TD3-full result shows large late upside but unsafe release. That is exactly the gap the distillation Markov SG-TD3 runner is designed to test.

The current distillation SG-TD3 Markov runner is therefore a scientifically clean next experiment:

- it keeps OF-MPC and LS available as the dynamic supervisor,
- it turns off live manually created safety layers,
- it keeps shadow diagnostics for audit,
- it uses 10 warm-start episodes and 3 critic-warm subepisodes,
- it lets SG-TD3 decide actor versus supervisor after critic warm.

## 29. Recommended Next Validation After A Real Distillation SG-TD3 Run

After the distillation SG-TD3 Markov run finishes, validate:

1. Confirm the config fields:
   - `agent_kind = "sg_td3"`
   - `notebook_source = "distillation_RL_assisted_MPC_markov_supervisor_gated_td3_critic_warm_unified.py"`
   - `markov_supervisor_mode = "ls_else_mpc"`
   - `markov_live_safety_mode = "shadow_only"`
   - `z_safety.enabled = False`
   - `td3_priority_fallback.enabled = False`
   - `markov_shadow_safety_enabled = True`

2. Plot post-warm and tail action-source fractions:
   - actor selected,
   - LS supervisor selected,
   - MPC supervisor selected,
   - fallback selected.

3. Compare the worst first 20 post-warm episodes against the TD3-full no-safeguard run.

4. Compare tail-20 reward and physical tracking against OF-MPC and TD3-full.

5. Check shadow projection logs:
   - how often old \(z\)-safety would have clipped,
   - whether those events correlate with low reward,
   - whether the SG gate avoided the worst high-risk actor actions.

6. Inspect \(z\)-norm and SG advantage:
   - early post-warm actor advantage should often be negative if the gate is protecting release,
   - tail actor advantage can become positive if the actor is truly useful.

## 30. Remaining Uncertainty

The main uncertainty is whether the TD3 critics in distillation will learn a reliable actor-versus-supervisor ranking fast enough. Distillation is more coupled and slower than polymer. The SG gate can be too conservative if the critics underrate the actor, or too permissive if the critics overrate risky actions.

The run should therefore be judged by both safety and learning:

- safety: no severe post-warm collapse,
- learning: increasing actor authority only when reward and tracking support it,
- control quality: tail tracking and input movement improve or remain competitive with OF-MPC,
- auditability: shadow diagnostics explain what the old safety layers would have changed.

The method is promising if it preserves the late upside of distillation TD3-full while removing or reducing the severe early release failures.
