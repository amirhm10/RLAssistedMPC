# 2026-04-20 Polymer Report Embedded Structured/ReID Refresh

- rewrote `report/polymer_change_impact_report.md` so the figures, tables, and explanation live inside the report instead of only linking out
- expanded `report/generate_polymer_change_impact_report.py` from residual+matrix only to residual, matrix, structured matrix, and reidentification
- added support for the second matrix rerun and the latest structured-matrix and reidentification runs
- added embedded analysis of why polymer reidentification is not working, using candidate-valid, update-success, fallback, condition-number, residual-ratio, eta, and source-code logs
- regenerated the polymer change-impact figures and CSV summaries under `report/polymer_change_impact/`
- included literature-backed discussion for running normalization, excitation/informative data windows, and ill-conditioned least-squares identification
