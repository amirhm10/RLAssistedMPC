from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SACAgent.supervisor_gated_sac_agent import SupervisorGatedSACAgent
from systems.polymer import get_polymer_notebook_defaults


def test_polymer_active_defaults_exclude_sg_sac_profiles():
    for family in ("weights", "residual", "markov"):
        nb = get_polymer_notebook_defaults(family)
        assert nb["agent_kind"] == "sg_td3"
        assert ("td3", "disturb") in nb["run_profiles"]
        assert ("sg_td3", "disturb") in nb["run_profiles"]
        assert ("sac", "disturb") not in nb["run_profiles"]
        assert ("sg_sac", "disturb") not in nb["run_profiles"]


def test_polymer_simple_continuous_runners_are_not_sg_sac_entrypoints():
    for filename in (
        "RL_assisted_MPC_weights_unified.py",
        "RL_assisted_MPC_residual_unified.py",
        "RL_assisted_MPC_markov_unified.py",
    ):
        source = (ROOT / filename).read_text(encoding="utf-8")
        assert "AGENT_KIND not in" in source
        assert "sg_td3" in source
        assert "sg_sac" not in source
        assert "SupervisorGatedSACAgent" not in source


def test_reusable_sg_sac_agent_package_remains_importable():
    assert SupervisorGatedSACAgent.__name__ == "SupervisorGatedSACAgent"


def run_direct():
    test_polymer_active_defaults_exclude_sg_sac_profiles()
    test_polymer_simple_continuous_runners_are_not_sg_sac_entrypoints()
    test_reusable_sg_sac_agent_package_remains_importable()
    print("supervisor_gated_sac_runner tests passed")


if __name__ == "__main__":
    run_direct()
