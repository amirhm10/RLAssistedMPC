from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_polymer_residual_sg_td3_profile_available():
    from systems.polymer import get_polymer_notebook_defaults

    nb = get_polymer_notebook_defaults("residual")
    assert ("sg_td3", "disturb") in nb["run_profiles"]
    assert ("sg_td3", "nominal") in nb["run_profiles"]
    assert "supervisor_gate" in nb


def test_residual_runner_imports_with_supervisor_gated_branch():
    from utils.residual_runner import run_residual_supervisor

    assert callable(run_residual_supervisor)


def run_direct():
    test_polymer_residual_sg_td3_profile_available()
    test_residual_runner_imports_with_supervisor_gated_branch()
    print("supervisor_gated_residual_integration tests passed")


if __name__ == "__main__":
    run_direct()
