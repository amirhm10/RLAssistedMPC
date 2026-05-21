---
name: slide-maker
description: Use this skill when the user asks to create, revise, audit, polish, or scientifically improve slides, Beamer presentations, talks, conference presentations, thesis slides, defense slides, research update slides, result slides, paper-to-slide summaries, or presentation figures. Trigger especially for slides, Beamer, LaTeX slides, presentation, talk, figures, plots, diagrams, flowcharts, results, journal comparison, paper figures, RL, MPC, Lyapunov, pH, polymer, distillation, CSTR, C2 splitter, or PhD-level research storytelling.
---

# Slide Maker Skill

Use this skill for academic research slides in the RL/MPC/process-control repositories. The goal is not only to make slides look nicer. The goal is to turn verified research evidence into a clear PhD-level oral story.

For result-based slides, use the `research-result-loop` mindset first: inspect the repository, identify the data and reports, reconstruct the method, check the claims, create or audit figures, then write the slide.

## Core identity

This is a research-to-Beamer skill.

Prioritize:

- scientific correctness
- claim-versus-evidence discipline
- readable figures
- clear mathematical notation
- reproducible result slides
- simple academic English
- the user's existing Beamer style

Do not prioritize decoration over evidence. Push back on slide text that overclaims what the data show.

## Case-study mindset

Treat these as related RL/process-control case studies:

- polymer CSTR control
- RL-assisted MPC with horizon, weight, model/matrix, residual, TD3, SAC, DQN, or dueling DQN agents
- direct Lyapunov MPC, safety gates, projection filters, fallback controllers, and target selectors
- distillation or C2 splitter control, including Aspen Dynamics workflows
- pH control or pH neutralization RL. Treat pH as another RL/process-control system. Slides should discuss pH dynamics, acid/base or buffer modeling, sensor reliability, flow-rate actions, reward design, constraints, setpoint tracking, and safety when relevant.

## Start with repository evidence

Before creating or editing slides, inspect what exists:

- Beamer decks: `StatsControl2026/`, `MACC2026/`, root `*.tex`, and other slide folders
- reports: `report/`, `reports/`, `change-reports/`
- figures: `figures/`, `Figures/`, `report/figures/`, `MACC2026/Figures/`, `StatsControl2026/figures/`
- results and data: `Results/`, `Result/`, `Data/`, `data/`, `outputs/`, `runs/`, `experiments/`
- notebooks and scripts used to generate figures
- papers: `papers/`, `Papers/`, `literature/`, `references/`, `pdfs/`, `PDFs/`, and local `.bib` files

If a location does not exist, do not invent it. Use what is actually in the repository.

## Preserve the user's slide style

When a local style exists, preserve it. In the current Stats and Control 2026 style, use these cues:

- Beamer 16:9
- `Copenhagen` theme
- maroon structure color
- muted blue, green, and red project colors
- compact blocks
- short scientific captions
- TikZ diagrams for method logic
- `takeawaybox` or an equivalent final message box
- project narrative: Project 1, Project 2, Project 3 when relevant

Do not change the global theme unless the user explicitly asks.

## Slide logic: claim, evidence, takeaway

For each technical slide, identify:

1. Claim: what the slide title says
2. Evidence: figure, equation, table, or diagram supporting the claim
3. Takeaway: what the audience should conclude

Prefer conclusion-style titles over topic-only titles.

Weak title:

`Project 3 Results`

Better title:

`The Lyapunov gate rejects actions mainly when the target is too far from the reachable region`

Then support it with evidence such as rejection frequency, target mismatch, or contraction residual plots.

## Recommended story structures

For a research update, use:

1. motivation
2. limitation of fixed MPC or naive RL
3. proposed RL/MPC role
4. mathematical formulation
5. implementation details
6. main results
7. interpretation
8. limitations
9. next experiment

For short talks, compress aggressively:

- 3 to 5 minutes: title, problem, method, one main result, takeaway
- 10 to 15 minutes: 8 to 12 main slides plus backup
- committee or defense update: 12 to 20 main slides plus backup

Do not copy a report paragraph into slides. Convert it into a spoken argument.

## Result-to-slide workflow

When slides depend on results:

1. Find the exact result files and reports.
2. Identify the baseline and proposed method.
3. Check what was held constant.
4. Compute or extract metrics when possible.
5. Use existing figures only after checking they match the claim.
6. Create new figures when existing figures are missing, unreadable, or not targeted to the slide claim.
7. State whether the plotted result is single-run, last episode, mean over seeds, or mean plus spread.
8. Separate training reward from evaluation tracking.
9. Avoid claiming improvement if the comparison is not fair.

Useful slide figures include:

- tracking plots with setpoints and tolerance bands
- manipulated inputs with bounds
- reward curves with smoothing and seed spread
- final-episode evaluation plots
- IAE, ISE, RMSE, final offset, overshoot, and settling summaries
- fallback rate, safety-gate acceptance, projection size, and Lyapunov residual plots
- target-selector plots comparing raw setpoint, admissible target, actual output, and `u_s`
- pH plots showing measured pH, reliable sensor channels, acid/base/water flow rates, setpoints, and prediction error

## Method-slide workflow

When the method is hard to explain with text, create a diagram.

Useful diagrams include:

- OF-MPC data generation, behavior cloning, and online TD3 fine-tuning
- RL-assisted MPC supervisor around a fixed MPC optimizer
- horizon, weight, model/matrix, and residual agent roles
- offset-free observer and MPC loop
- direct Lyapunov target and safety gate
- RL propose, gate certify, MPC fallback
- replay-buffer design and mixed sampling
- pH mixing and sensor-feedback loop

Use TikZ when exact labels and LaTeX notation matter. Use Python-generated PDF, SVG, or PNG when data or geometry is easier to manage outside LaTeX.

## Figure quality rules

Before placing a figure on a slide, check:

- Does it support the slide claim?
- Are axes, units, setpoints, bounds, and legends readable?
- Are fonts large enough for projection?
- Are line styles distinguishable in grayscale?
- Is the crop tight enough?
- Are labels consistent with the manuscript or report?
- Is the figure generated from the correct data file?
- Is a zoomed tail plot needed to show offset or settling?

Do not hide poor performance. If results are mixed, present them as mixed and explain the next experiment.

## Paper and citation workflow

Use papers carefully.

Search in this order:

1. local paper folders if present
2. local BibTeX files
3. user-provided PDFs or reports
4. verified online sources when web access is available

When citing a paper, explain what role it plays:

- MPC baseline or industrial MPC motivation
- RL process-control motivation
- safe RL
- offline RL
- residual RL
- value-augmented MPC
- reward-shaping issues
- pH neutralization or buffer-modeling context

Do not invent citations. Do not claim a paper supports something unless the source was checked.

## Beamer editing rules

When editing an existing deck:

- preserve the theme, colors, notation, and bibliography style
- keep each slide to one main message
- reduce word-heavy frames
- keep equations only when they help the oral explanation
- move dense derivations to backup slides
- use macros for repeated notation where practical
- avoid broad rewrites unless requested
- avoid fragile spacing hacks unless needed for layout
- compile after significant edits when possible
- inspect the rendered PDF pages, not only the TeX source

If helper scripts exist, use them when helpful:

- `scripts/audit_beamer.py` for missing images, heavy frames, placeholders, and long lines
- `scripts/collect_slide_assets.py` for finding slide inputs
- `scripts/check_figures.py` for figure dimensions and naming issues
- PDF-to-image conversion or visual inspection tools when available

If these scripts do not exist in the current repository, proceed manually and do not pretend they were run.

## Visual review workflow

For serious slide edits:

1. Compile the deck if possible.
2. Convert or inspect the PDF pages visually when possible.
3. Check for clipping, tiny fonts, overcrowding, missing figures, bad crops, and unreadable legends.
4. Iterate until the slide is presentation-ready.

If compilation or visual inspection is not possible, report that clearly.

## Output summary for slide tasks

At the end of a slide task, report:

1. Files inspected
2. Slide story or method used
3. Figures created, reused, or audited
4. Files changed
5. Compilation or validation performed
6. Remaining issues
7. Best next improvement

## Preservation rules

- Do not delete old slides, figures, notebooks, reports, or raw data unless explicitly asked.
- Prefer creating a new version over overwriting important work.
- Do not fabricate results, citations, or paper claims.
- Do not make the slides stronger than the evidence.
- Keep edits minimal, traceable, and consistent with the user's style.
- Use clear academic English.
- Use LaTeX for equations.
- Avoid hard-to-copy special symbols in prose.
- Do not use semicolons in prose.
