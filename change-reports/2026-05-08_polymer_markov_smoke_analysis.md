# Polymer Markov Smoke-Run Analysis

## Summary

Extended the polymer Markov correction progress report with a quantitative review of the existing 30-step smoke run.

## Analysis Assets

New analysis tables and figures were generated under:

`Polymer/Results/polymer_markov_corrected_mpc/20260508_115807/`

The added assets include output-wise tracking metrics, input-movement summaries, acceptance and prediction-score summaries, candidate-selection counts, executed correction statistics, and mechanism-level diagnostic figures.

## Main Observation

The smoke run verifies implementation mechanics and lifted-prediction equivalence, but it does not prove controller improvement. Viscosity tracking improved slightly, temperature tracking worsened slightly, all eligible steps accepted corrections, and several correction coefficients frequently hit their bounds.

## Verification

The analysis was generated from the saved `input_data.pkl` bundle and checked against the written CSV summaries. The report now references the generated tables and figures.
