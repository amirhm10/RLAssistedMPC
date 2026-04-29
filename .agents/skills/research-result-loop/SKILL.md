---
name: research-result-loop
description: Use this skill automatically for any task involving RL/MPC research analysis, result interpretation, report writing, LaTeX report updates, mathematical derivations, figure creation or auditing, citation support, literature connection, experiment comparison, next-step planning, or summarizing what worked and what failed. Trigger this skill for words or phrases such as report, analysis, analyze, results, figures, plots, paper, citation, literature, math, derivation, next step, experiment, compare, reward, MPC, RL, Lyapunov, residual policy, replay buffer, model identification, distillation, polymer, CSTR, or column.
---

# Research Result Loop Skill

Use this skill whenever the user asks for research-stage work, especially for RL, MPC, Lyapunov filters, residual policies, reward shaping, replay buffers, model identification, process-control case studies, result analysis, figure review, report writing, or literature-supported interpretation.

The goal is not only to edit code. The goal is to behave like a careful researcher: reconstruct the method, verify the implementation, interpret the results scientifically, connect the findings to the literature, update the report, and propose the next experiment.

## Repository-specific locations

After inspecting the repository, list the key locations here if they exist:

- Reports: `report/`, `change-reports/`, `MACC2026/`
- Figures: `report/figures/`, `report/polymer_change_impact/figures/`, `report/polymer_wide_range_matrix_structured/figures/`
- Results: `Polymer/Results/`, `Distillation/Results/`, `Result/`, `Data/`
- Notebooks: root `*.ipynb`, including `RL_assisted_MPC_*_unified.ipynb`, `distillation_*_unified.ipynb`, `MPCOffsetFree_unified.ipynb`, `systemIdentification.ipynb`
- Source code: `Simulation/`, `systems/`, `TD3Agent/`, `SACAgent/`, `DQN/`, `DuelingDQN/`
- Shared modules: `utils/`, `systems/distillation/`, `BasicFunctions/`
- Papers or references: `MACC2026/`, `report/`
- BibTeX files: none found in repo inspection
- Case-study folders: `Polymer/`, `Distillation/`, `VanDeVusse/`

If a location is unknown, do not invent it.

## Core research loop

For every research-result-loop task, follow this order.

### 1. Understand the current experiment

Identify the relevant case study, method, and files.

Check whether the task concerns:

- Polymer CSTR
- C2 splitter or distillation column
- Offset-free MPC
- MPC-pretrained RL
- TD3, SAC, DQN, or dueling DQN
- Lyapunov or safety projection filter
- Residual policy
- Reward shaping
- Replay buffer or mixed replay
- Prioritized experience replay
- Model identification or re-identification
- Weight tuning or matrix-multiplier methods
- Report writing or paper revision
- Figure generation or figure auditing

Do not infer conclusions from filenames only. Inspect the code, saved results, metrics, plots, and report text when available.

### 2. Reconstruct the method mathematically

Before interpreting results, write the mathematical structure of the method.

Include the relevant items when applicable:

- State vector
- Output vector
- Manipulated input vector
- Setpoint definition
- Scaled deviation coordinates versus physical coordinates
- Input constraints
- Output constraints
- Observer equations
- Offset-free augmentation
- MPC optimization problem
- Target selector or steady-state optimization
- RL state
- RL action
- Reward function
- Replay-buffer sampling rule
- Safety filter or projection problem
- Model-identification update rule

Use LaTeX for equations in reports and mathematical explanations.

Be explicit about notation. Do not mix physical and scaled variables without saying so.

### 3. Verify implementation consistency

Actively look for scientific and coding inconsistencies.

Check for:

- Sign errors in rewards
- Incorrect reward scaling
- Wrong use of physical versus scaled variables
- Wrong setpoint indexing
- Wrong input bounds
- Wrong disturbance or observer update
- Wrong done flag
- Mismatch between logged reward and stored reward
- Mismatch between training reward and plotted reward
- Replay-buffer sampling bias
- PER priority update mistakes
- Warm-start or frozen-actor logic issues
- Actor output scaling mistakes
- Incorrect use of delta_u versus absolute u
- Inconsistent random seeds
- Figure generated from the wrong result file
- Report claims that are stronger than the actual results

If something looks suspicious, state exactly where it appears and why it matters.

### 4. Analyze results quantitatively

Do not rely only on visual impressions.

When data are available, compute or report metrics such as:

- IAE
- ISE
- RMSE
- Maximum absolute error
- Steady-state offset
- Settling time
- Overshoot
- Constraint violations
- Input movement
- Move suppression
- Reward components
- Final tracking error
- Per-setpoint performance
- Per-episode learning trend
- Comparison before and after online fine-tuning

Separate transient performance from near-setpoint performance.

If the controller improves tracking but receives a worse reward, investigate whether the reward is misaligned with the evaluation objective.

### 5. Audit figures

For every figure used in a report or analysis, check:

- Does the figure support the claim?
- Are setpoints visible?
- Are units shown?
- Are legends clear?
- Are line styles distinguishable?
- Are warm-start regions shown clearly?
- Are constraints or tolerance bands shown when relevant?
- Are zoomed near-setpoint views needed?
- Are panels consistent across methods?
- Are colors and labels consistent with other figures?
- Is the figure generated from the correct result file?

Do not overwrite old figures. Save revised figures using clear new filenames unless the user explicitly requests replacement.

### 6. Connect to literature

Do not invent citations.

Use only:

- Papers already present in the repo
- BibTeX files in the repo
- PDFs or references provided by the user
- Sources that can be verified online if web access is available

When citing a paper, explain what role it plays. For example:

- Supports MPC-RL integration
- Supports safe RL
- Supports offline RL
- Supports reward shaping concerns
- Supports residual RL
- Supports value-augmented MPC
- Supports model adaptation or re-identification
- Supports process-control application context

If adding a citation to LaTeX, check that the citation key exists in the BibTeX file. If it does not exist, add a proper BibTeX entry only when the source is verified.

### 7. Update reports scientifically

When writing or revising report text, use a simple academic tone.

Avoid exaggerated claims.

Use this structure when appropriate:

- Objective
- Method
- Mathematical formulation
- Experimental setup
- Results
- Interpretation
- Limitations
- Next experiment

Clearly distinguish:

- What was tested
- What was observed
- What the observation likely means
- What remains uncertain
- What should be tested next

Do not make the report sound like the method is proven if only simulation evidence is available.

### 8. Propose next experiments

Next steps must be concrete.

For each proposed next experiment, include:

- Purpose
- Exact file or module likely involved
- What to change
- What metric should improve
- What failure mode to watch for
- What figure should be generated
- What result would confirm or reject the idea

Avoid vague suggestions such as "tune the reward more" unless you specify exactly what parameter, why, and how to evaluate it.

## Required response format for research tasks

When completing a research-result-loop task, respond using this structure:

1. Files inspected
2. What the current method is doing
3. Mathematical interpretation
4. Main result interpretation
5. Bugs, inconsistencies, or risks found
6. Figure/report updates made
7. Literature connections
8. Recommended next experiment
9. Remaining uncertainty

If code or report files were changed, also include:

10. Files changed
11. How to verify the changes

## Writing style

Use a scientific but clear tone.

Prefer direct explanations over fancy wording.

Avoid unnecessary rewriting of the user's report. Make targeted edits unless the user asks for a full rewrite.

Use LaTeX for equations.

Do not use hard-to-copy special symbols. Prefer ASCII text in normal prose.

Do not use semicolons in prose.

## Preservation rules

- Do not delete raw results.
- Do not overwrite old figures unless explicitly asked.
- Do not rewrite notebooks broadly unless required.
- Do not refactor unrelated code.
- Do not change the scientific meaning of the user's report without explaining why.
- Keep changes minimal, traceable, and testable.
- If results are inconclusive, say so clearly.
- If a citation cannot be verified, do not use it as evidence.
- If a plot or metric is missing, explain what is missing and how to generate it.
