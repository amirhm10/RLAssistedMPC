# Polymer Residual And Matrix Change-Impact Report

Date: 2026-04-20

## Summary

Added a focused polymer-only analysis comparing the latest refreshed residual and matrix runs against their pre-change references and the disturbance baseline MPC.

## Outputs

- [report/polymer_change_impact_report.md](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/report/polymer_change_impact_report.md)
- [report/generate_polymer_change_impact_report.py](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/report/generate_polymer_change_impact_report.py)
- figures and CSV tables under [report/polymer_change_impact/](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/report/polymer_change_impact)

## Main Result

- the refreshed polymer residual run shows a clear improvement versus the legacy residual run
- the refreshed polymer matrix run also changes behavior and improves the aggregate tail metrics, but the gain is smaller and more mixed
- the analysis explicitly notes that the saved refreshed runs were produced before the later observer-default rollback, so they still reflect the current-measurement-corrector observer path
