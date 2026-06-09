from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SACAgent.supervisor_gated_sac_agent import SupervisorGatedSACAgent
from systems.distillation import get_distillation_notebook_defaults, resolve_distillation_agent_kind
from systems.distillation.notebook_params import DISTILLATION_NOTEBOOK_DEFAULTS


ACTIVE_DISTILLATION_RL_FAMILIES = ("horizon_standard", "markov", "weights", "residual", "combined")
CONTINUOUS_FAMILIES = ("markov", "weights", "residual")


def test_distillation_active_defaults_exclude_sac_td7_and_dueling_profiles():
    horizon = get_distillation_notebook_defaults("horizon_standard")
    assert horizon["agent_mode"] == "sg"
    assert horizon["agent_kind"] == "sg_dqn"
    assert ("dqn", "disturb", "fluctuation") in horizon["run_profiles"]
    assert ("sg_dqn", "disturb", "fluctuation") in horizon["run_profiles"]
    assert all(key[0] in {"dqn", "sg_dqn"} for key in horizon["run_profiles"])

    for family in CONTINUOUS_FAMILIES:
        nb = get_distillation_notebook_defaults(family)
        assert nb["agent_mode"] == "sg"
        assert nb["agent_kind"] == "sg_td3"
        assert ("td3", "disturb", "fluctuation") in nb["run_profiles"]
        assert ("sg_td3", "disturb", "fluctuation") in nb["run_profiles"]
        assert all(key[0] in {"td3", "sg_td3"} for key in nb["run_profiles"])
        assert ("sac", "disturb", "fluctuation") not in nb["run_profiles"]
        assert ("sg_sac", "disturb", "fluctuation") not in nb["run_profiles"]
        assert ("td7", "disturb", "fluctuation") not in nb["run_profiles"]


def test_distillation_plain_mode_resolves_to_non_sg_agents():
    assert resolve_distillation_agent_kind("horizon", "plain") == "dqn"
    assert resolve_distillation_agent_kind("markov", "plain") == "td3"
    assert resolve_distillation_agent_kind("weights", "without_sg") == "td3"
    assert resolve_distillation_agent_kind("residual", "no-sg") == "td3"


def test_distillation_active_default_table_excludes_archived_families():
    assert set(DISTILLATION_NOTEBOOK_DEFAULTS) == {
        "system_identification",
        "baseline",
        "horizon_standard",
        "markov",
        "weights",
        "residual",
        "combined",
    }
    for archived_family in (
        "horizon_dueling",
        "matrix",
        "structured_matrix",
        "reidentification",
    ):
        try:
            get_distillation_notebook_defaults(archived_family)
        except KeyError:
            pass
        else:
            raise AssertionError(f"{archived_family} should not be an active distillation default")


def test_distillation_active_runners_are_not_sac_td7_or_dueling_entrypoints():
    for filename in (
        "distillation_RL_assisted_MPC_horizons_unified.py",
        "distillation_RL_assisted_MPC_markov_unified.py",
        "distillation_RL_assisted_MPC_weights_unified.py",
        "distillation_RL_assisted_MPC_residual_unified.py",
    ):
        source = (ROOT / filename).read_text(encoding="utf-8")
        assert "resolve_distillation_agent_kind" in source
        assert "AGENT_MODE" in source
        assert "SupervisorGatedSACAgent" not in source
        assert "SACAgent" not in source
        assert "sg_sac" not in source
        assert "td7" not in source.lower()
        assert "dueling" not in source.lower()


def test_reusable_sg_sac_agent_package_remains_importable():
    assert SupervisorGatedSACAgent.__name__ == "SupervisorGatedSACAgent"


def run_direct():
    test_distillation_active_defaults_exclude_sac_td7_and_dueling_profiles()
    test_distillation_plain_mode_resolves_to_non_sg_agents()
    test_distillation_active_default_table_excludes_archived_families()
    test_distillation_active_runners_are_not_sac_td7_or_dueling_entrypoints()
    test_reusable_sg_sac_agent_package_remains_importable()
    print("distillation active runner cleanup tests passed")


if __name__ == "__main__":
    run_direct()
