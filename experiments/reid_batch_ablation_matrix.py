from __future__ import annotations

from copy import deepcopy


SHORT_DIAGNOSTIC_PRESET = {
    "n_tests": 40,
    "set_points_len": 200,
    "warm_start": 10,
    "test_cycle": [False, False, False, False, False],
}


def _eta_label(force_eta_constant):
    if force_eta_constant is None:
        return "eta_learned"
    return f"eta_{float(force_eta_constant):.2f}".replace(".", "p")


def _score_row(row: dict, metric_key: str = "tail_reward_mean") -> float:
    try:
        return float(row.get(metric_key, float("-inf")))
    except (TypeError, ValueError):
        return float("-inf")


def _is_tied(score_a: float, score_b: float) -> bool:
    tol = max(0.5, 0.05 * max(abs(score_a), abs(score_b), 1.0))
    return abs(float(score_a) - float(score_b)) <= tol


def _base_fixed_cfg(study_defaults: dict) -> dict:
    return {
        "agent_kind": str(study_defaults["agent_kind"]).lower(),
        "run_mode": str(study_defaults["run_mode"]).lower(),
        "state_mode": str(study_defaults["state_mode"]).lower(),
    }


def build_tier1_run_specs(study_defaults: dict) -> list[dict]:
    study_cfg = deepcopy(study_defaults["study"])
    tier_cfg = study_cfg["tier1"]
    fixed = dict(_base_fixed_cfg(study_defaults))
    fixed.update(dict(tier_cfg["fixed"]))
    preset = dict(study_cfg["short_diagnostic_preset"])
    specs: list[dict] = []
    for basis_family in tier_cfg["basis_families"]:
        for id_component_mode in tier_cfg["id_component_modes"]:
            for case in tier_cfg["cases"]:
                disable_identification = bool(case["disable_identification"])
                force_eta_constant = case["force_eta_constant"]
                case_label = "noid" if disable_identification else "id"
                study_label = f"t1_{basis_family}_{id_component_mode}_{case_label}_{_eta_label(force_eta_constant)}"
                spec = {
                    **fixed,
                    **preset,
                    "study_label": study_label,
                    "ablation_group": "tier1",
                    "basis_family": basis_family,
                    "id_component_mode": id_component_mode,
                    "disable_identification": disable_identification,
                    "force_eta_constant": force_eta_constant,
                    "lambda_prev_A": 1e-2,
                    "lambda_prev_B": 1e-1,
                    "lambda_0_A": 1e-4,
                    "lambda_0_B": 1e-3,
                    "theta_low_A": -0.15,
                    "theta_high_A": 0.15,
                    "theta_low_B": -0.08,
                    "theta_high_B": 0.08,
                }
                specs.append(spec)
    return specs


def _best_rows_by_basis(rows: list[dict]) -> dict[str, dict]:
    best: dict[str, dict] = {}
    for row in rows:
        basis = str(row["basis_family"])
        if basis not in best or _score_row(row) > _score_row(best[basis]):
            best[basis] = row
    return best


def select_best_tier1_basis_family(tier1_rows: list[dict]) -> str:
    if not tier1_rows:
        raise ValueError("Tier 1 summary rows are required to select a basis family.")
    best_rows = _best_rows_by_basis(tier1_rows)
    return max(best_rows.values(), key=_score_row)["basis_family"]


def select_tier2_component_modes(tier1_rows: list[dict], best_basis_family: str) -> list[str]:
    basis_rows = [row for row in tier1_rows if str(row["basis_family"]) == str(best_basis_family)]
    if not basis_rows:
        raise ValueError(f"No tier1 rows found for basis family {best_basis_family}.")
    best_by_mode: dict[str, dict] = {}
    for row in basis_rows:
        mode = str(row["id_component_mode"])
        if mode not in best_by_mode or _score_row(row) > _score_row(best_by_mode[mode]):
            best_by_mode[mode] = row
    ranked_modes = sorted(best_by_mode.values(), key=_score_row, reverse=True)
    top_mode = str(ranked_modes[0]["id_component_mode"])
    selected = [top_mode]
    mode_scores = {str(row["id_component_mode"]): _score_row(row) for row in ranked_modes}
    if "A_only" in mode_scores and "AB" in mode_scores and _is_tied(mode_scores["A_only"], mode_scores["AB"]):
        selected = sorted(set(selected + ["A_only", "AB"]))
    return selected


def build_tier2_run_specs(study_defaults: dict, tier1_rows: list[dict]) -> list[dict]:
    study_cfg = deepcopy(study_defaults["study"])
    tier_cfg = study_cfg["tier2"]
    fixed = dict(_base_fixed_cfg(study_defaults))
    fixed.update(dict(study_cfg["tier1"]["fixed"]))
    preset = dict(study_cfg["short_diagnostic_preset"])
    best_basis_family = select_best_tier1_basis_family(tier1_rows)
    selected_modes = select_tier2_component_modes(tier1_rows, best_basis_family)
    specs: list[dict] = []
    for id_component_mode in selected_modes:
        for regime_name, regime_cfg in tier_cfg["parameter_regimes"].items():
            for force_eta_constant in tier_cfg["eta_cases"]:
                study_label = (
                    f"t2_{best_basis_family}_{id_component_mode}_{regime_name}_{_eta_label(force_eta_constant)}"
                )
                specs.append(
                    {
                        **fixed,
                        **preset,
                        **regime_cfg,
                        "study_label": study_label,
                        "ablation_group": "tier2",
                        "basis_family": best_basis_family,
                        "id_component_mode": id_component_mode,
                        "disable_identification": False,
                        "force_eta_constant": force_eta_constant,
                        "regime_name": regime_name,
                    }
                )
    return specs


def select_best_row(rows: list[dict], groups: tuple[str, ...] = ("tier1", "tier2")) -> dict:
    filtered = [row for row in rows if str(row.get("ablation_group")) in groups]
    if not filtered:
        raise ValueError("No summary rows available for best-row selection.")
    return max(filtered, key=_score_row)


def build_tier3_run_specs(study_defaults: dict, prior_rows: list[dict]) -> list[dict]:
    study_cfg = deepcopy(study_defaults["study"])
    tier_cfg = study_cfg["tier3"]
    fixed = dict(_base_fixed_cfg(study_defaults))
    fixed.update(dict(study_cfg["tier1"]["fixed"]))
    preset = dict(study_cfg["short_diagnostic_preset"])
    best_row = select_best_row(prior_rows, groups=("tier1", "tier2"))
    best_basis_family = str(best_row["basis_family"])
    best_component_mode = str(best_row["id_component_mode"])
    best_regime = {
        "lambda_prev_A": float(best_row["lambda_prev_A"]),
        "lambda_prev_B": float(best_row["lambda_prev_B"]),
        "lambda_0_A": float(best_row["lambda_0_A"]),
        "lambda_0_B": float(best_row["lambda_0_B"]),
        "theta_low_A": float(best_row["theta_low_A"]),
        "theta_high_A": float(best_row["theta_high_A"]),
        "theta_low_B": float(best_row["theta_low_B"]),
        "theta_high_B": float(best_row["theta_high_B"]),
    }
    specs: list[dict] = []
    for observer_update_alignment in tier_cfg["observer_update_alignment"]:
        for normalize_blend_extras in tier_cfg["normalize_blend_extras"]:
            for force_eta_constant in tier_cfg["eta_cases"]:
                study_label = (
                    f"t3_{best_basis_family}_{best_component_mode}_{observer_update_alignment}_"
                    f"norm_{int(bool(normalize_blend_extras))}_{_eta_label(force_eta_constant)}"
                )
                specs.append(
                    {
                        **fixed,
                        **preset,
                        **best_regime,
                        "study_label": study_label,
                        "ablation_group": "tier3",
                        "basis_family": best_basis_family,
                        "id_component_mode": best_component_mode,
                        "disable_identification": False,
                        "force_eta_constant": force_eta_constant,
                        "observer_update_alignment": observer_update_alignment,
                        "normalize_blend_extras": bool(normalize_blend_extras),
                        "regime_name": str(best_row.get("regime_name", "best_prior")),
                    }
                )
    return specs
