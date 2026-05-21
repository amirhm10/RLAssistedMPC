---
name: research-result-loop
description: Use this skill automatically for deep RL/MPC/process-control research analysis, result interpretation, report writing, LaTeX or Markdown report updates, mathematical derivations, figure creation or auditing, literature connection, experiment comparison, and next-step planning. Trigger this skill for report, analysis, results, figures, plots, paper, citation, literature, math, derivation, next experiment, compare, reward, MPC, RL, Lyapunov, safety gate, residual policy, replay buffer, model identification, target selector, pH, neutralization, polymer, CSTR, distillation, C2 splitter, column, or what worked and failed.
---

# Research Result Loop Skill

Use this skill whenever the user asks for research-stage work in RL, MPC, safe RL, Lyapunov filters, residual policies, reward shaping, replay buffers, model identification, process-control case studies, result analysis, figure review, report writing, paper comparison, or literature-supported interpretation.

The goal is not only to edit code or summarize outputs. The goal is to behave like a careful researcher: reconstruct the method, verify the implementation, analyze raw data, create or audit figures, interpret the results scientifically, connect the findings to papers, update the report, and propose the next experiment.

## Active repositories and case-study mindset

Apply this skill across the current RL/process-control repositories, including:

- Polymer CSTR RL/MPC and Lyapunov safety studies.
- RL-assisted MPC studies with horizon, weight, model/matrix, residual, DQN, TD3, or SAC agents.
- Distillation or C2 splitter studies, including Aspen Dynamics or large-scale column workflows.
- pH control or pH neutralization RL studies. Treat the pH system as another RL/process-control case study, not as a generic chemistry-only task. Analyze pH dynamics, acid/base or buffer modeling, measurement reliability, flow-rate actions, reward design, constraints, setpoint tracking, and safety in the same way as the other RL systems.

Do not infer conclusions from filenames only. Inspect code, reports, saved data, metrics, and figures whenever available.

## Repository locations to check

First scan the repository and use the locations that actually exist. Common locations may include:

- Reports: `report/`, `reports/`, `change-reports/`, `MACC2026/`, `StatsControl2026/`.
- Figures: `figures/`, `Figures/`, `report/figures/`, `MACC2026/Figures/`, `StatsControl2026/figures/`.
- Results and data: `Results/`, `Result/`, `Data/`, `data/`, `outputs/`, `runs/`, `experiments/`.
- Notebooks: root `*.ipynb` and task-specific notebook folders.
- Source code: `Simulation/`, `systems/`, `utils/`, `TD3Agent/`, `SACAgent/`, `DQN/`, `DuelingDQN/`, `agents/`, `envs/`, `controllers/`.
- Papers and references: `papers/`, `Papers/`, `literature/`, `references/`, `pdfs/`, `PDFs/`, `ref_lib.bib`, `acs-main.bib`, or any local `.bib` file.

If a location is unknown, do not invent it. State what was found and what was missing.

## Mandatory research workflow

For every research-result-loop task, follow this order.

### 1. Read the history and define the experiment

Identify:

- the case study and plant
- the controller or agent
- the baseline
- the proposed change
- the data files
- the report or change-report history
- the figures already used
- what was previously tried and what failed

Do not suggest an idea that the local reports already show was tried, unless the new version is clearly different.

### 2. Reconstruct the method mathematically

Before interpreting results, write the mathematical structure of the method.

Include the relevant items when applicable:

- state vector
- output vector
- manipulated input vector
- setpoint definition
- physical coordinates versus scaled deviation coordinates
- input and output constraints
- observer or estimator equations
- offset-free augmentation
- MPC optimization problem
- target selector or steady-state optimization
- RL state and action
- action scaling or clipping
- reward function and reward components
- replay-buffer sampling rule
- TD3, SAC, DQN, or other learning update
- Lyapunov function, contraction test, projection, or safety gate
- model-identification or re-identification update rule
- pH model assumptions, such as acid/base balance, Henderson-Hasselbalch approximation, buffer behavior, sensor choice, and flow mixing, when the task is about the pH system

Use LaTeX for equations in reports and mathematical explanations. Be explicit about notation. Do not mix physical and scaled variables without saying so.

### 3. Verify implementation consistency

Actively look for scientific and coding inconsistencies.

Check for:

- sign errors in rewards
- incorrect reward scaling
- wrong use of physical versus scaled variables
- wrong setpoint indexing
- wrong input bounds or action mapping
- wrong pH sensor column, stream mapping, or concentration assumption in pH tasks
- wrong disturbance or observer update
- wrong done flag
- mismatch between logged reward and stored reward
- mismatch between training reward and plotted reward
- replay-buffer sampling bias
- PER priority update mistakes
- warm-start or frozen-actor logic issues
- actor output scaling mistakes
- incorrect use of delta_u versus absolute u
- inconsistent random seeds
- figure generated from the wrong result file
- report claims that are stronger than the actual results

If something looks suspicious, state exactly where it appears and why it matters.

### 4. Analyze results quantitatively

Do not rely only on visual impressions.

When data are available, compute or report metrics such as:

- IAE, ISE, RMSE, and maximum absolute error
- final or steady-state offset
- settling time and overshoot
- constraint violations
- input movement and saturation
- move suppression
- reward components
- final tracking error
- per-setpoint performance
- per-episode learning trend
- accepted versus rejected RL actions
- fallback frequency
- projection correction size
- Lyapunov contraction residual
- target-selector slack or target mismatch
- pH setpoint error, pH overshoot, buffer-region behavior, flow-rate usage, and sensor-noise sensitivity for pH tasks

Separate transient performance from near-setpoint performance. Separate training reward from evaluation tracking. If reward improves but tracking worsens, investigate reward misalignment.

### 5. Support the analysis with figures

Every major claim should be supported by one of the following:

- an existing figure that is verified to match the data and claim
- a new figure generated from raw results
- a clear statement that the needed data are missing

When raw data or saved bundles are available, create or audit figures such as:

- output tracking with setpoints and tolerance bands
- zoomed tail plots for offset and settling
- manipulated-input trajectories with bounds
- delta_u trajectories
- saturation and constraint-activity plots
- reward versus episode with smoothing and seed spread
- actor/critic losses if available
- replay-buffer composition or PER diagnostics
- Lyapunov value and contraction residual plots
- accepted RL action versus fallback action plots
- projection correction size plots
- target-selector plots showing raw setpoint, admissible target, actual output, `u_s`, target slack, and target feasibility
- pH plots showing measured pH, reliable sensor channels, acid/base/water flow rates, setpoints, and model prediction error
- comparison figures for OF-MPC, RL_1, RL_2, residual, weight, horizon, model/matrix, cold-start, and pretrained cases

Prefer a small set of high-signal figures over many weak figures. Do not hide poor performance. Do not compare methods unless the setup is fair.

When creating a figure, save:

- the figure file
- the script or notebook cell used to generate it when practical
- the source data path
- the metric definitions
- a short note explaining what claim the figure supports

Use a dated folder such as `report/figures/YYYY-MM-DD_short_task_name/` when appropriate. Do not overwrite old figures unless explicitly asked.

### 6. Diagnose mechanisms and failure modes

For control-focused analysis, separate:

- raw setpoint tracking versus modified-target tracking
- steady-target quality versus closed-loop tracking quality
- candidate-controller quality versus safety-filter correction quality
- nominal-case performance versus disturbed-case performance
- true controller improvement versus easier target definition
- Lyapunov feasibility improvement versus softened fallback or slack behavior

For target-selector and Lyapunov work, explicitly check:

- whether the raw setpoint is reachable under input bounds
- whether `y_sp`, admissible `y_s`, and actual `y` differ
- whether `d_hat` makes equality target selection too restrictive
- whether `x_hat` is near zero while `d_hat` absorbs bias
- whether `u_s` is stuck because of bounds, regularization, or objective weights
- whether Lyapunov rejection is caused by the RL action, the target, the model, or the contraction rate
- whether fallback behavior hides infeasibility

### 7. Connect to literature and papers

Do not invent citations.

Search in this order:

1. local paper folders if present: `papers/`, `Papers/`, `literature/`, `references/`, `pdfs/`, `PDFs/`
2. local BibTeX files
3. user-provided PDFs or reports
4. verified online sources when web access is available

When citing a paper, explain its role. For example:

- supports MPC-RL integration
- supports safe RL
- supports offline RL
- supports reward-shaping concerns
- supports residual RL
- supports value-augmented MPC
- supports model adaptation or re-identification
- supports pH neutralization, titration, buffer modeling, or process-control application context

If adding a citation to LaTeX, check that the citation key exists. If it does not exist, add a proper BibTeX entry only when the source is verified.

### 8. Update reports scientifically

When writing or revising report text, use a simple academic tone.

Use this structure when appropriate:

- Objective
- Method
- Mathematical formulation
- Experimental setup
- Results
- Figures and evidence
- Interpretation
- Limitations
- Next experiment

Clearly distinguish:

- what was tested
- what was observed
- what the observation likely means
- what remains uncertain
- what should be tested next

Do not make the report sound like the method is proven if only simulation evidence or limited lab data are available.

### 9. Propose next experiments

Next steps must be concrete. For each proposed experiment, include:

- purpose
- exact file or module likely involved
- what to change
- what metric should improve
- what failure mode to watch for
- what figure should be generated
- what result would confirm or reject the idea

Avoid vague suggestions such as "tune the reward more" unless you specify exactly what parameter, why, and how to evaluate it.

## Required response format

When completing a research-result-loop task, respond using this structure:

1. Files inspected
2. What the current method is doing
3. Mathematical interpretation
4. Figures or data evidence used
5. Main result interpretation
6. Bugs, inconsistencies, or risks found
7. Literature connections
8. Recommended next experiment
9. Remaining uncertainty

If files were changed, also include:

10. Files changed
11. How to verify the changes

## Writing and preservation rules

- Use a scientific but clear tone.
- Prefer direct explanations over fancy wording.
- Avoid unnecessary rewriting of the user's report.
- Make targeted edits unless the user asks for a full rewrite.
- Use LaTeX for equations.
- Do not use hard-to-copy special symbols in prose. Prefer ASCII text.
- Do not use semicolons in prose.
- Do not delete raw results.
- Do not overwrite old figures unless explicitly asked.
- Do not broadly rewrite notebooks unless required. Treat notebooks as structured analysis artifacts.
- Prefer extracting reusable analysis into scripts when creating figures or metrics.
- Do not refactor unrelated code.
- Keep changes minimal, traceable, and testable.
- If results are inconclusive, say so clearly.
- If a citation cannot be verified, do not use it as evidence.
- If a plot or metric is missing, explain what is missing and how to save it next time.
