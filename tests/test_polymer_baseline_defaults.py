from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from systems.polymer import get_polymer_notebook_defaults


def test_polymer_baseline_disturb_profile_matches_active_rl_episode_schedule():
    baseline = get_polymer_notebook_defaults("baseline")
    horizon = get_polymer_notebook_defaults("horizon_standard")
    markov = get_polymer_notebook_defaults("markov")
    weights = get_polymer_notebook_defaults("weights")
    residual = get_polymer_notebook_defaults("residual")
    combined = get_polymer_notebook_defaults("combined")

    disturb = baseline["run_profiles"]["disturb"]
    expected_episode = horizon["episode_defaults"]

    assert baseline["run_mode"] == "disturb"
    assert baseline["warm_start_override"] is None
    assert disturb["use_disturbance"] is True
    assert disturb["profile_name"] == expected_episode["profile_name"] == "robustness_200_100"
    assert disturb["n_tests"] == expected_episode["n_tests"] == 300
    assert disturb["set_points_len"] == expected_episode["set_points_len"] == 400
    assert disturb["warm_start"] == expected_episode["warm_start"] == 10
    assert disturb["test_cycle"] == expected_episode["test_cycle"]

    for nb in (markov, weights, residual, combined):
        assert nb["episode_defaults"]["profile_name"] == disturb["profile_name"]
        assert nb["episode_defaults"]["n_tests"] == disturb["n_tests"]
        assert nb["episode_defaults"]["set_points_len"] == disturb["set_points_len"]
        assert nb["episode_defaults"]["warm_start"] == disturb["warm_start"]
        assert nb["episode_defaults"]["test_cycle"] == disturb["test_cycle"]


def test_polymer_baseline_uses_shared_setpoint_and_default_mode_in_script():
    source = (ROOT / "MPCOffsetFree_unified.py").read_text(encoding="utf-8")

    assert 'RUN_MODE = NB["run_mode"]' in source
    assert "y_sp_scenario_phys = SYS[\"rl_setpoints_phys\"].copy()" in source
    assert 'RUN_MODE = "nominal"' not in source
    assert "np.array([[2., 326.0], [3.4, 321.0]]" not in source


def run_direct():
    test_polymer_baseline_disturb_profile_matches_active_rl_episode_schedule()
    test_polymer_baseline_uses_shared_setpoint_and_default_mode_in_script()
    print("polymer baseline default tests passed")


if __name__ == "__main__":
    run_direct()
