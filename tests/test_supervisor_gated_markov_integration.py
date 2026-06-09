from __future__ import annotations

import inspect
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TD3Agent.supervisor_gated_agent import SupervisorGatedTD3Agent
from TD3Agent.supervisor_replay_buffer import SOURCE_POLICY
from systems.polymer import get_polymer_notebook_defaults
from utils.markov_runner import (
    get_markov_rl_state_dim,
    make_td3_markov_agent,
    resolve_markov_agent_state_features,
    resolve_markov_supervisor_action,
    run_single_closed_loop,
)


def test_polymer_markov_sg_td3_profile_available():
    nb = get_polymer_notebook_defaults("markov")
    assert nb["agent_kind"] == "sg_td3"
    assert nb["run_mode"] == "disturb"
    assert nb["state_mode"] == "mismatch"
    assert ("sg_td3", "disturb") in nb["run_profiles"]
    assert ("sg_td3", "nominal") in nb["run_profiles"]
    assert ("sg_sac", "disturb") not in nb["run_profiles"]
    assert nb["markov_supervisor_mode"] == "ls_else_mpc"
    assert "supervisor_gate" in nb


def test_polymer_sg_td3_markov_defaults_shadow_only():
    configured = get_polymer_notebook_defaults("markov")
    ctrl = configured["controller"]
    bc = configured["behavioral_cloning"]

    assert configured["agent_kind"] == "sg_td3"
    assert configured["run_mode"] == "disturb"
    assert configured["state_mode"] == "mismatch"
    assert configured["episode_defaults"]["warm_start"] == 10
    assert configured["post_warm_start_action_freeze_subepisodes"] == 3
    assert configured["post_warm_start_actor_freeze_subepisodes"] == 3
    assert configured["markov_supervisor_mode"] == "ls_else_mpc"
    assert configured["markov_live_safety_mode"] == "shadow_only"
    assert "mismatch" in configured["run_profiles"][("sg_td3", "disturb")]["result_prefix"]
    assert bc["enabled"] is False
    assert bc["handoff"]["enabled"] is False
    assert ctrl["z_safety"]["enabled"] is False
    assert ctrl["z_bound"] == 0.50
    assert ctrl["td3_priority_fallback"]["enabled"] is False
    assert ctrl["td3_authority_ramp"]["enabled"] is False
    assert ctrl["rl_fallback_to_ls"] is False
    assert ctrl["force_td3_respects_warm_start"] is True
    assert ctrl["markov_shadow_safety"]["enabled"] is True
    assert ctrl["markov_shadow_safety"]["compute_ls_candidate"] is False
    assert configured["supervisor_gate"]["advantage_margin"] == 0.0
    assert configured["supervisor_gate"]["score_uncertainty_weight"] == 0.5
    assert configured["supervisor_gate"]["score_supervisor_action_weight"] == 0.02
    assert configured["supervisor_gate"]["score_previous_action_weight"] == 0.01


def test_make_td3_markov_agent_returns_supervisor_gated_agent():
    configured = get_polymer_notebook_defaults("markov")
    agent = make_td3_markov_agent(configured, state_dim=12, action_dim=4, set_points_len=20)
    assert isinstance(agent, SupervisorGatedTD3Agent)


def test_markov_standard_state_dimension_excludes_mismatch_features():
    standard_dim = get_markov_rl_state_dim(
        base_aug_dim=10,
        n_outputs=2,
        n_inputs=2,
        z_dim=4,
        state_mode="standard",
    )
    mismatch_dim = get_markov_rl_state_dim(
        base_aug_dim=10,
        n_outputs=2,
        n_inputs=2,
        z_dim=4,
        state_mode="mismatch",
    )

    assert standard_dim == 10 + 2 + 2 + 4 + 4 + 2
    assert mismatch_dim == standard_dim + 2 * 2


def test_markov_base_only_standard_state_dimension_excludes_markov_features():
    standard_full_dim = get_markov_rl_state_dim(
        base_aug_dim=10,
        n_outputs=2,
        n_inputs=2,
        z_dim=4,
        state_mode="standard",
    )
    standard_base_only_dim = get_markov_rl_state_dim(
        base_aug_dim=10,
        n_outputs=2,
        n_inputs=2,
        z_dim=4,
        state_mode="standard",
        markov_agent_state_features="base_only",
    )
    mismatch_base_only_dim = get_markov_rl_state_dim(
        base_aug_dim=10,
        n_outputs=2,
        n_inputs=2,
        z_dim=4,
        state_mode="mismatch",
        markov_agent_state_features="base_only",
    )

    assert resolve_markov_agent_state_features({"markov_agent_state_features": "standard_standard"}) == "base_only"
    assert standard_full_dim == 10 + 2 + 2 + 4 + 4 + 2
    assert standard_base_only_dim == 10 + 2 + 2
    assert mismatch_base_only_dim == standard_base_only_dim + 2 * 2


def test_polymer_markov_simple_runner_is_td3_only():
    source = (ROOT / "RL_assisted_MPC_markov_unified.py").read_text(encoding="utf-8")

    assert "AGENT_KIND not in" in source
    assert "sg_td3" in source
    assert "sg_sac" not in source


def test_standard_markov_mode_still_computes_observer_innovation():
    source = inspect.getsource(run_single_closed_loop)
    innovation_assignment = 'innovation = history["y_scaled_dev"][step, :] - yhat'
    observer_update = "x_model = A @ x_model + B @ u_dev + L @ innovation"

    assert innovation_assignment in source
    assert observer_update in source
    assert source.index(innovation_assignment) < source.index(observer_update)


def test_resolve_markov_supervisor_action_ls_else_mpc():
    z_ls = np.asarray([0.01, -0.02, 0.03, -0.04], dtype=float)
    ls_payload = resolve_markov_supervisor_action(
        mode="ls_else_mpc",
        z_ls=z_ls,
        ls_accepted=True,
        z_bound=0.05,
        z_dim=4,
    )
    assert ls_payload["kind_name"] == "ls"
    assert np.allclose(ls_payload["z"], z_ls)
    assert np.allclose(ls_payload["raw_action"], z_ls / 0.05)

    mpc_payload = resolve_markov_supervisor_action(
        mode="ls_else_mpc",
        z_ls=z_ls,
        ls_accepted=False,
        z_bound=0.05,
        z_dim=4,
    )
    assert mpc_payload["kind_name"] == "mpc"
    assert np.allclose(mpc_payload["z"], np.zeros(4))
    assert np.allclose(mpc_payload["raw_action"], np.zeros(4))


def test_supervisor_gated_markov_replay_metadata_roundtrip():
    configured = get_polymer_notebook_defaults("markov")
    agent = make_td3_markov_agent(configured, state_dim=6, action_dim=4, set_points_len=20)
    state = np.zeros(6, dtype=np.float32)
    next_state = np.ones(6, dtype=np.float32) * 0.1
    action = np.asarray([0.1, -0.1, 0.2, -0.2], dtype=np.float32)
    supervisor = np.zeros(4, dtype=np.float32)
    previous = np.ones(4, dtype=np.float32) * 0.05

    agent.push_supervised(
        state,
        action,
        -1.0,
        next_state,
        False,
        policy_action=action,
        supervisor_action=supervisor,
        previous_action=previous,
        selected_source=SOURCE_POLICY,
        score_policy=1.25,
        score_supervisor=1.0,
        advantage_policy_supervisor=0.25,
    )
    batch = agent.buffer.sample_supervised(1, device=agent.device)
    assert batch["policy_actions"].shape == (1, 4)
    assert batch["supervisor_actions"].shape == (1, 4)
    assert int(batch["selected_sources"][0].item()) == SOURCE_POLICY


def run_direct():
    test_polymer_markov_sg_td3_profile_available()
    test_polymer_sg_td3_markov_defaults_shadow_only()
    test_make_td3_markov_agent_returns_supervisor_gated_agent()
    test_markov_standard_state_dimension_excludes_mismatch_features()
    test_markov_base_only_standard_state_dimension_excludes_markov_features()
    test_polymer_markov_simple_runner_is_td3_only()
    test_standard_markov_mode_still_computes_observer_innovation()
    test_resolve_markov_supervisor_action_ls_else_mpc()
    test_supervisor_gated_markov_replay_metadata_roundtrip()
    print("supervisor_gated_markov_integration tests passed")


if __name__ == "__main__":
    run_direct()
