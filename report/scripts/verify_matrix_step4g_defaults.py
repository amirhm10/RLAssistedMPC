from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from systems.distillation.config import (
    DISTILLATION_BASELINE_RUN_PROFILES,
    DISTILLATION_COMBINED_RUN_PROFILES,
    DISTILLATION_HORIZON_RUN_PROFILES,
    DISTILLATION_MATRIX_RUN_PROFILES,
    DISTILLATION_REIDENTIFICATION_RUN_PROFILES,
    DISTILLATION_RESIDUAL_RUN_PROFILES,
    DISTILLATION_WEIGHT_RUN_PROFILES,
)
from systems.distillation.notebook_params import (
    DISTILLATION_COMBINED_DEFAULTS,
    DISTILLATION_HORIZON_DUELING_DEFAULTS,
    DISTILLATION_HORIZON_STANDARD_DEFAULTS,
    DISTILLATION_MATRIX_DEFAULTS,
    DISTILLATION_REIDENTIFICATION_DEFAULTS,
    DISTILLATION_RESIDUAL_DEFAULTS,
    DISTILLATION_STRUCTURED_MATRIX_DEFAULTS,
    DISTILLATION_WEIGHT_DEFAULTS,
)
from systems.polymer.notebook_params import (
    POLYMER_MATRIX_DEFAULTS,
    POLYMER_STRUCTURED_MATRIX_DEFAULTS,
)


def _assert_equal(actual, expected, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def _assert_false(value: bool, label: str) -> None:
    _assert_equal(bool(value), False, label)


def _assert_true(value: bool, label: str) -> None:
    _assert_equal(bool(value), True, label)


def _normalize_profiles(profiles: dict) -> list[tuple[tuple[str, ...], dict]]:
    normalized = {}
    for key, value in profiles.items():
        tuple_key = key if isinstance(key, tuple) else (key,)
        if len(tuple_key) == 3:
            _, run_mode, disturbance_profile = tuple_key
        elif len(tuple_key) == 2:
            run_mode, disturbance_profile = tuple_key
        else:
            raise AssertionError(f"Unexpected run profile key shape: {tuple_key!r}")
        collapsed_key = (run_mode, disturbance_profile)
        profile = dict(value)
        if collapsed_key in normalized and normalized[collapsed_key] != profile:
            raise AssertionError(f"Inconsistent duplicate run profile for {collapsed_key!r}")
        normalized[collapsed_key] = profile
    return sorted(normalized.items(), key=lambda item: item[0])


def _check_matrix_defaults() -> None:
    polymer_expected = [
        ("polymer scalar", POLYMER_MATRIX_DEFAULTS, 2, 2, 0.3, 20, 8, 12),
        ("polymer structured", POLYMER_STRUCTURED_MATRIX_DEFAULTS, 3, 3, 0.6, 25, 10, 15),
    ]
    for label, cfg, action_freeze, actor_freeze, lambda_bc_start, active_subepisodes, protected, ramp in polymer_expected:
        bc = cfg["behavioral_cloning"]
        ctrl = cfg["controller"]
        caps = ctrl["release_protected_advisory_caps"]
        _assert_true(bc["enabled"], f"{label} behavioral_cloning.enabled")
        _assert_true(caps["enabled"], f"{label} release_protected_advisory_caps.enabled")
        _assert_false(ctrl["mpc_acceptance_fallback"]["enabled"], f"{label} mpc_acceptance_fallback.enabled")
        _assert_false(ctrl["mpc_dual_cost_shadow"]["enabled"], f"{label} mpc_dual_cost_shadow.enabled")
        _assert_false(ctrl["mpc_usefulness_gate"]["enabled"], f"{label} mpc_usefulness_gate.enabled")
        _assert_equal(cfg["post_warm_start_action_freeze_subepisodes"], action_freeze, f"{label} action freeze")
        _assert_equal(cfg["post_warm_start_actor_freeze_subepisodes"], actor_freeze, f"{label} actor freeze")
        _assert_equal(bc["lambda_bc_start"], lambda_bc_start, f"{label} lambda_bc_start")
        _assert_equal(bc["active_subepisodes"], active_subepisodes, f"{label} active_subepisodes")
        _assert_equal(caps["protected_live_subepisodes"], protected, f"{label} protected_live_subepisodes")
        _assert_equal(caps["authority_ramp_subepisodes"], ramp, f"{label} authority_ramp_subepisodes")

    structured_labels = POLYMER_STRUCTURED_MATRIX_DEFAULTS["behavioral_cloning"]["label_weight_overrides"]
    if not structured_labels:
        raise AssertionError("polymer structured label_weight_overrides should remain populated")

    distillation_expected = [
        ("distillation scalar", DISTILLATION_MATRIX_DEFAULTS),
        ("distillation structured", DISTILLATION_STRUCTURED_MATRIX_DEFAULTS),
    ]
    for label, cfg in distillation_expected:
        bc = cfg["behavioral_cloning"]
        ctrl = cfg["controller"]
        caps = ctrl["release_protected_advisory_caps"]
        _assert_true(bc["enabled"], f"{label} behavioral_cloning.enabled")
        _assert_true(caps["enabled"], f"{label} release_protected_advisory_caps.enabled")
        _assert_false(ctrl["mpc_acceptance_fallback"]["enabled"], f"{label} mpc_acceptance_fallback.enabled")
        _assert_false(ctrl["mpc_dual_cost_shadow"]["enabled"], f"{label} mpc_dual_cost_shadow.enabled")
        _assert_false(ctrl["mpc_usefulness_gate"]["enabled"], f"{label} mpc_usefulness_gate.enabled")
        _assert_equal(cfg["post_warm_start_action_freeze_subepisodes"], 5, f"{label} action freeze")
        _assert_equal(cfg["post_warm_start_actor_freeze_subepisodes"], 5, f"{label} actor freeze")
        _assert_equal(bc["lambda_bc_start"], 0.3, f"{label} lambda_bc_start")
        _assert_equal(bc["active_subepisodes"], 20, f"{label} active_subepisodes")
        _assert_equal(caps["protected_live_subepisodes"], 15, f"{label} protected_live_subepisodes")
        _assert_equal(caps["authority_ramp_subepisodes"], 30, f"{label} authority_ramp_subepisodes")
        _assert_equal(bc["label_weight_overrides"], {}, f"{label} label_weight_overrides")


def _check_distillation_run_profiles() -> None:
    expected_rl_profiles = _normalize_profiles(DISTILLATION_HORIZON_RUN_PROFILES)
    rl_profile_tables = {
        "horizon config": DISTILLATION_HORIZON_RUN_PROFILES,
        "horizon defaults": DISTILLATION_HORIZON_STANDARD_DEFAULTS["run_profiles"],
        "dueling defaults": DISTILLATION_HORIZON_DUELING_DEFAULTS["run_profiles"],
        "matrix defaults": DISTILLATION_MATRIX_DEFAULTS["run_profiles"],
        "structured matrix defaults": DISTILLATION_STRUCTURED_MATRIX_DEFAULTS["run_profiles"],
        "weights defaults": DISTILLATION_WEIGHT_DEFAULTS["run_profiles"],
        "weights config": DISTILLATION_WEIGHT_RUN_PROFILES,
        "residual defaults": DISTILLATION_RESIDUAL_DEFAULTS["run_profiles"],
        "residual config": DISTILLATION_RESIDUAL_RUN_PROFILES,
        "reidentification defaults": DISTILLATION_REIDENTIFICATION_DEFAULTS["run_profiles"],
        "reidentification config": DISTILLATION_REIDENTIFICATION_RUN_PROFILES,
        "combined defaults": DISTILLATION_COMBINED_DEFAULTS["run_profiles"],
        "combined config": DISTILLATION_COMBINED_RUN_PROFILES,
    }
    for label, profiles in rl_profile_tables.items():
        _assert_equal(_normalize_profiles(profiles), expected_rl_profiles, f"{label} run_profiles")

    nominal_baseline = DISTILLATION_BASELINE_RUN_PROFILES[("nominal", "none")]
    rl_nominal = dict(DISTILLATION_HORIZON_RUN_PROFILES[("nominal", "none")])
    if nominal_baseline == rl_nominal:
        raise AssertionError("distillation baseline nominal profile should remain separate from RL defaults")


def _check_distillation_notebook_defaults() -> None:
    rl_defaults = {
        "horizon": DISTILLATION_HORIZON_STANDARD_DEFAULTS,
        "dueling": DISTILLATION_HORIZON_DUELING_DEFAULTS,
        "matrix": DISTILLATION_MATRIX_DEFAULTS,
        "structured_matrix": DISTILLATION_STRUCTURED_MATRIX_DEFAULTS,
        "weights": DISTILLATION_WEIGHT_DEFAULTS,
        "residual": DISTILLATION_RESIDUAL_DEFAULTS,
        "reidentification": DISTILLATION_REIDENTIFICATION_DEFAULTS,
        "combined": DISTILLATION_COMBINED_DEFAULTS,
    }
    override_keys = [
        "n_tests_override",
        "set_points_len_override",
        "warm_start_override",
        "test_cycle_override",
        "plot_start_episode_override",
        "compare_start_episode_override",
    ]
    for label, cfg in rl_defaults.items():
        for key in override_keys:
            _assert_equal(cfg[key], None, f"{label} {key}")


def _check_distillation_notebook_sources() -> None:
    expected = {
        "distillation_RL_assisted_MPC_horizons_unified.ipynb",
        "distillation_RL_assisted_MPC_horizons_dueling_unified.ipynb",
        "distillation_RL_assisted_MPC_matrices_unified.ipynb",
        "distillation_RL_assisted_MPC_structured_matrices_unified.ipynb",
        "distillation_RL_assisted_MPC_weights_unified.ipynb",
        "distillation_RL_assisted_MPC_residual_unified.ipynb",
        "distillation_RL_assisted_MPC_reidentification_unified.ipynb",
        "distillation_RL_assisted_MPC_combined_unified.ipynb",
    }
    for notebook_name in sorted(expected):
        text = (REPO_ROOT / notebook_name).read_text(encoding="utf-8")
        if 'NB[\\"run_profiles\\"]' not in text:
            raise AssertionError(f"{notebook_name} should read NB[\"run_profiles\"]")


if __name__ == "__main__":
    _check_matrix_defaults()
    _check_distillation_run_profiles()
    _check_distillation_notebook_defaults()
    _check_distillation_notebook_sources()
    print("Matrix Step 4G default verification passed.")
