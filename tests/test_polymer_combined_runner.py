from __future__ import annotations

import pathlib
import sys
from types import SimpleNamespace

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TD3Agent.supervisor_replay_buffer import SOURCE_POLICY, SOURCE_SUPERVISOR
from systems.polymer import get_polymer_notebook_defaults, resolve_polymer_combined_agent_kinds
from utils.agent_step_runtime import (
    replay_train_supervisor_gated_continuous_agent,
    select_supervisor_gated_continuous_action,
)
from utils.combined_runner import _float_or_nan
from utils.phase1_hidden_release import ACTION_SOURCE_PHASE1_HIDDEN_BASELINE
from utils.residual_authority import project_residual_action


def test_polymer_combined_defaults_use_all_sg_mismatch_mode():
    nb = get_polymer_notebook_defaults("combined")
    markov_nb = get_polymer_notebook_defaults("markov")

    assert nb["combined_agent_mode"] == "sg"
    assert nb["run_mode"] == "disturb"
    assert nb["enable_horizon"] is True
    assert nb["enable_markov"] is True
    assert nb["enable_weights"] is True
    assert nb["enable_residual"] is True
    assert nb["enable_matrix"] is False

    assert resolve_polymer_combined_agent_kinds(nb["combined_agent_mode"]) == {
        "horizon_agent_kind": "sg_dqn",
        "markov_agent_kind": "sg_td3",
        "weights_agent_kind": "sg_td3",
        "residual_agent_kind": "sg_td3",
    }
    assert nb["horizon_state_mode"] == "mismatch"
    assert nb["markov_state_mode"] == "mismatch"
    assert nb["weights_state_mode"] == "mismatch"
    assert nb["residual_state_mode"] == "mismatch"
    assert nb["horizon_post_warm_start_action_freeze_subepisodes"] == 3
    assert nb["td3_post_warm_start_action_freeze_subepisodes"] == 3
    assert nb["td3_post_warm_start_actor_freeze_subepisodes"] == 3

    assert nb["horizon_supervisor_gate"]["advantage_margin"] == 0.0
    assert nb["markov_supervisor_gate"]["advantage_margin"] == 0.0
    assert nb["weights_supervisor_gate"]["advantage_margin"] == 0.5
    assert nb["residual_supervisor_gate"]["advantage_margin"] == 0.5
    assert nb["controller"]["z_bound"] == markov_nb["controller"]["z_bound"] == 0.50
    assert nb["controller"]["z_safety"]["enabled"] is False
    assert nb["weight_safety"]["fallback_to_identity_on_nonfinite"] is True
    assert nb["residual_safety"]["fallback_to_zero_on_nonfinite"] is True
    assert nb["append_rho_to_state"] is False
    assert nb["residual_authority_enabled"] is False
    assert nb["authority_use_rho"] is False
    assert nb["residual_zero_deadband_enabled"] is False


def test_polymer_combined_plain_mode_resolves_all_plain_agents():
    assert resolve_polymer_combined_agent_kinds("plain") == {
        "horizon_agent_kind": "dqn",
        "markov_agent_kind": "td3",
        "weights_agent_kind": "td3",
        "residual_agent_kind": "td3",
    }


def test_polymer_combined_root_runner_is_active_sg_plain_only():
    source = (ROOT / "RL_assisted_MPC_combined_unified.py").read_text(encoding="utf-8")

    assert "resolve_polymer_combined_agent_kinds" in source
    assert "SupervisorGatedDQNAgent" in source
    assert "SupervisorGatedTD3Agent" in source
    assert '"weight_safety": dict(NB.get("weight_safety", {}))' in source
    assert '"residual_safety": dict(NB.get("residual_safety", {}))' in source
    assert "residual_authority_enabled" in source
    assert "SACAgent" not in source
    assert "DuelingDQN" not in source
    assert "TD7" not in source
    assert "legacy matrix branch disabled" in source

    runtime_source = (ROOT / "utils" / "combined_runner.py").read_text(encoding="utf-8")
    assert "decision_interval=1" in runtime_source
    assert "SG_SOURCE_FALLBACK" in runtime_source
    assert "markov_sg_solver_fallback_source_fraction" in runtime_source


class _FakeSGContinuousAgent:
    def __init__(self):
        self.calls = []

    def select_action_with_supervisor(self, state, supervisor_action, previous_action=None, explore=False, test=False):
        self.calls.append((state, supervisor_action, previous_action, explore, test))
        policy = np.asarray([0.4, -0.2], dtype=np.float32)
        supervisor = np.asarray(supervisor_action, dtype=np.float32)
        previous = np.asarray(previous_action, dtype=np.float32)
        return SimpleNamespace(
            action=policy.copy(),
            policy_action=policy.copy(),
            supervisor_action=supervisor.copy(),
            selected_source=SOURCE_POLICY,
            score_policy=2.0,
            score_supervisor=1.0,
            advantage_policy_supervisor=1.0,
            q1_policy=2.1,
            q2_policy=1.9,
            q1_supervisor=1.1,
            q2_supervisor=0.9,
            q_gap_policy=0.2,
            q_gap_supervisor=0.2,
            previous_action=previous,
        )


def test_sg_continuous_hidden_window_executes_supervisor_and_records_policy():
    phase1 = {
        "enabled": True,
        "first_live_action_step": 5,
        "hidden_window_active_log": np.array([0, 0, 1, 1, 0, 0], dtype=int),
    }
    baseline = np.zeros(2, dtype=np.float32)

    decision = select_supervisor_gated_continuous_action(
        agent=_FakeSGContinuousAgent(),
        state=np.zeros(3, dtype=np.float32),
        step=2,
        warm_start_step=1,
        test=False,
        baseline_action=baseline,
        supervisor_action=baseline,
        phase1=phase1,
        action_dim=2,
    )

    assert np.allclose(decision.action, baseline)
    assert np.allclose(decision.policy_action, [0.4, -0.2])
    assert decision.source == ACTION_SOURCE_PHASE1_HIDDEN_BASELINE
    assert decision.selected_source == SOURCE_SUPERVISOR
    assert decision.decision_taken == 0
    assert decision.advantage_policy_supervisor == 1.0


def test_sg_continuous_replay_helper_pushes_supervised_metadata():
    class RecorderAgent:
        def __init__(self):
            self.calls = []
            self.train_calls = 0

        def push_supervised(self, *args, **kwargs):
            self.calls.append((args, kwargs))

        def train_step(self, bc_context=None):
            self.train_calls += 1
            return {"critic_updated": True, "actor_slot": False}

    agent = RecorderAgent()
    decision = select_supervisor_gated_continuous_action(
        agent=_FakeSGContinuousAgent(),
        state=np.zeros(3, dtype=np.float32),
        step=6,
        warm_start_step=1,
        test=False,
        baseline_action=np.zeros(2, dtype=np.float32),
        supervisor_action=np.zeros(2, dtype=np.float32),
        phase1={"enabled": False, "first_live_action_step": 2, "hidden_window_active_log": np.zeros(8, dtype=int)},
        action_dim=2,
    )

    replay_train_supervisor_gated_continuous_agent(
        agent=agent,
        state=np.zeros(3, dtype=np.float32),
        action=np.asarray([0.1, 0.2], dtype=np.float32),
        reward=-1.0,
        next_state=np.ones(3, dtype=np.float32),
        done=0.0,
        step=6,
        test=False,
        train_start_step=1,
        decision=decision,
    )

    assert agent.train_calls == 1
    args, kwargs = agent.calls[0]
    assert np.allclose(args[1], [0.1, 0.2])
    assert np.allclose(kwargs["policy_action"], [0.4, -0.2])
    assert np.allclose(kwargs["supervisor_action"], [0.0, 0.0])
    assert int(kwargs["selected_source"]) == SOURCE_POLICY


def test_combined_residual_rho_logging_handles_authority_disabled_projection():
    projection = project_residual_action(
        action_raw=np.zeros(2, dtype=np.float32),
        low_coef=np.asarray([-0.2, -0.2], dtype=np.float32),
        high_coef=np.asarray([0.2, 0.2], dtype=np.float32),
        u_base=np.asarray([0.5, 0.5], dtype=np.float32),
        scaled_current_input=np.asarray([0.5, 0.5], dtype=np.float32),
        u_min_scaled_abs=np.zeros(2, dtype=np.float32),
        u_max_scaled_abs=np.ones(2, dtype=np.float32),
        apply_authority=False,
        authority_use_rho=False,
    )

    assert projection["rho"] is None
    assert np.isnan(_float_or_nan(projection["rho"]))
    assert np.isnan(_float_or_nan(projection["rho_raw"]))
    assert np.isnan(_float_or_nan(projection["rho_eff"]))


def run_direct():
    test_polymer_combined_defaults_use_all_sg_mismatch_mode()
    test_polymer_combined_plain_mode_resolves_all_plain_agents()
    test_polymer_combined_root_runner_is_active_sg_plain_only()
    test_sg_continuous_hidden_window_executes_supervisor_and_records_policy()
    test_sg_continuous_replay_helper_pushes_supervised_metadata()
    test_combined_residual_rho_logging_handles_authority_disabled_projection()
    print("polymer combined runner tests passed")


if __name__ == "__main__":
    run_direct()
