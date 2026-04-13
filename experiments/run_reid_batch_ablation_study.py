from __future__ import annotations

import csv
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from SACAgent.sac_agent import SACAgent
from Simulation.mpc import MpcSolverGeneral
from Simulation.system_functions import PolymerCSTR
from TD3Agent.agent import TD3Agent
from experiments.reid_batch_ablation_matrix import (
    SHORT_DIAGNOSTIC_PRESET,
    build_tier1_run_specs,
    build_tier2_run_specs,
    build_tier3_run_specs,
)
from systems.polymer import (
    POLYMER_OBSERVER_POLES,
    POLYMER_SYSTEM_METADATA,
    get_polymer_notebook_defaults,
    load_polymer_system_data,
)
from utils.helpers import apply_min_max
from utils.notebook_setup import prepare_polymer_notebook_env
from utils.plotting import plot_reid_batch_ablation_summary, plot_reid_batch_theta_diagnostics
from utils.reid_batch import get_blend_state_dim_v2, summarize_reid_run_statistics
from utils.reid_batch_runner import run_reid_batch_supervisor
from utils.rewards import make_reward_fn_relative_QR


def _set_global_seed(seed: int | None) -> None:
    if seed is None:
        return
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _timestamped_dir(parent: Path, prefix_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = parent / prefix_name / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _resolve_runtime_roots(study_defaults: dict, repo_root: str | Path | None = None) -> tuple[Path, Path, Path]:
    if repo_root is None:
        repo_root, data_dir, result_dir = prepare_polymer_notebook_env(
            data_dir_override=study_defaults.get("data_dir_override"),
            results_dir_override=study_defaults.get("results_dir_override"),
        )
    else:
        repo_root = Path(repo_root).resolve()
        _, data_dir, result_dir = prepare_polymer_notebook_env(
            data_dir_override=study_defaults.get("data_dir_override"),
            results_dir_override=study_defaults.get("results_dir_override"),
        )
    return Path(repo_root), Path(data_dir), Path(result_dir)


def _build_static_polymer_context(study_defaults: dict, repo_root: Path):
    sys_cfg = study_defaults["system_setup"]
    system_params = sys_cfg["system_params"].copy()
    system_design_params = sys_cfg["design_params"].copy()
    system_steady_state_inputs = sys_cfg["ss_inputs"].copy()
    delta_t = float(sys_cfg["delta_t_hours"])

    cstr_ss = PolymerCSTR(system_params, system_design_params, system_steady_state_inputs, delta_t)
    steady_states = {"ss_inputs": cstr_ss.ss_inputs, "y_ss": cstr_ss.y_ss}

    input_bounds = sys_cfg["input_bounds"]
    system_data = load_polymer_system_data(
        repo_root,
        steady_states=steady_states,
        setpoint_y=sys_cfg["setpoint_range_phys"].copy(),
        u_min=input_bounds["u_min"].copy(),
        u_max=input_bounds["u_max"].copy(),
        n_inputs=2,
        data_override=study_defaults.get("data_dir_override"),
    )

    reward_params, reward_fn = make_reward_fn_relative_QR(
        system_data["data_min"],
        system_data["data_max"],
        int(system_data["B_aug"].shape[1]),
        **study_defaults["reward"],
    )
    y_sp_scenario_phys = sys_cfg["rl_setpoints_phys"].copy()
    inputs_number = int(system_data["B_aug"].shape[1])
    y_sp_scenario = (
        apply_min_max(y_sp_scenario_phys, system_data["data_min"][inputs_number:], system_data["data_max"][inputs_number:])
        - apply_min_max(steady_states["y_ss"], system_data["data_min"][inputs_number:], system_data["data_max"][inputs_number:])
    )
    return {
        "system_params": system_params,
        "system_design_params": system_design_params,
        "system_steady_state_inputs": system_steady_state_inputs,
        "delta_t": delta_t,
        "steady_states": steady_states,
        "system_data": system_data,
        "reward_params": reward_params,
        "reward_fn": reward_fn,
        "y_sp_scenario": y_sp_scenario,
    }


def _resolve_replay_settings(agent_cfg: dict, set_points_len: int) -> dict:
    buffer_size = int(agent_cfg["buffer_size"])
    recent_window = (
        int(agent_cfg["replay_recent_window"])
        if agent_cfg["replay_recent_window"] is not None
        else min(buffer_size, int(agent_cfg["replay_recent_window_mult"]) * int(set_points_len))
    )
    return {
        "buffer_size": buffer_size,
        "replay_frac_per": float(agent_cfg["replay_frac_per"]),
        "replay_frac_recent": float(agent_cfg["replay_frac_recent"]),
        "replay_recent_window": int(recent_window),
        "replay_alpha": float(agent_cfg["replay_alpha"]),
        "replay_beta_start": float(agent_cfg["replay_beta_start"]),
        "replay_beta_end": float(agent_cfg["replay_beta_end"]),
        "replay_beta_steps": int(agent_cfg["replay_beta_steps"]),
    }


def _build_reid_cfg_and_runtime(
    study_defaults: dict,
    static_ctx: dict,
    run_spec: dict,
) -> tuple[dict, dict]:
    agent_kind = str(run_spec.get("agent_kind", study_defaults["agent_kind"])).lower()
    run_mode = str(run_spec.get("run_mode", study_defaults["run_mode"])).lower()
    state_mode = str(run_spec.get("state_mode", study_defaults["state_mode"])).lower()
    episode_cfg = study_defaults["episode_defaults"]
    controller_cfg = study_defaults["controller"]
    reid_defaults = study_defaults["reid"]
    agent_cfg = study_defaults["td3_agent"] if agent_kind == "td3" else study_defaults["sac_agent"]
    replay_cfg = _resolve_replay_settings(agent_cfg, int(run_spec.get("set_points_len", episode_cfg["set_points_len"])))

    n_tests = int(run_spec.get("n_tests", episode_cfg["n_tests"]))
    set_points_len = int(run_spec.get("set_points_len", episode_cfg["set_points_len"]))
    warm_start = int(run_spec.get("warm_start", episode_cfg["warm_start"]))
    test_cycle = list(run_spec.get("test_cycle", episode_cfg["test_cycle"]))
    seed = int(run_spec.get("seed", study_defaults.get("seed", 42)))

    system = PolymerCSTR(
        static_ctx["system_params"].copy(),
        static_ctx["system_design_params"].copy(),
        static_ctx["system_steady_state_inputs"].copy(),
        static_ctx["delta_t"],
    )
    system_data = static_ctx["system_data"]
    A_aug = np.asarray(system_data["A_aug"], float)
    B_aug = np.asarray(system_data["B_aug"], float)
    C_aug = np.asarray(system_data["C_aug"], float)
    A_phys = np.asarray(system_data["A"], float)
    B_phys = np.asarray(system_data["B"], float)
    C_phys = np.asarray(system_data["C"], float)
    n_inputs = int(B_aug.shape[1])
    n_outputs = int(C_aug.shape[0])
    state_dim = int(get_blend_state_dim_v2(A_aug.shape[0], n_outputs, n_inputs, state_mode))
    action_dim = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mpc_obj = MpcSolverGeneral(
        A_aug,
        B_aug,
        C_aug,
        Q_out=np.array([controller_cfg["Q1_penalty"], controller_cfg["Q2_penalty"]], float),
        R_in=np.array([controller_cfg["R1_penalty"], controller_cfg["R2_penalty"]], float),
        NP=int(controller_cfg["predict_h"]),
        NC=int(controller_cfg["cont_h"]),
    )

    if agent_kind == "td3":
        agent = TD3Agent(
            state_dim=state_dim,
            action_dim=action_dim,
            actor_hidden=list(agent_cfg["actor_hidden"]),
            critic_hidden=list(agent_cfg["critic_hidden"]),
            gamma=agent_cfg["gamma"],
            actor_lr=agent_cfg["actor_lr"],
            critic_lr=agent_cfg["critic_lr"],
            batch_size=agent_cfg["batch_size"],
            policy_delay=agent_cfg["policy_delay"],
            target_policy_smoothing_noise_std=agent_cfg["target_policy_smoothing_noise_std"],
            noise_clip=agent_cfg["noise_clip"],
            max_action=agent_cfg["max_action"],
            tau=agent_cfg["tau"],
            std_start=agent_cfg["std_start"],
            std_end=agent_cfg["std_end"],
            std_decay_rate=agent_cfg["std_decay_rate"],
            std_decay_mode=agent_cfg["std_decay_mode"],
            buffer_size=replay_cfg["buffer_size"],
            replay_frac_per=replay_cfg["replay_frac_per"],
            replay_frac_recent=replay_cfg["replay_frac_recent"],
            replay_recent_window=replay_cfg["replay_recent_window"],
            replay_alpha=replay_cfg["replay_alpha"],
            replay_beta_start=replay_cfg["replay_beta_start"],
            replay_beta_end=replay_cfg["replay_beta_end"],
            replay_beta_steps=replay_cfg["replay_beta_steps"],
            device=device,
            actor_freeze=agent_cfg["actor_freeze"],
            exploration_mode=agent_cfg["exploration_mode"],
            loss_type=agent_cfg["loss_type"],
            param_noise_resample_interval=agent_cfg["param_noise_resample_interval"],
            n_step=int(agent_cfg["n_step"]),
            multistep_mode=agent_cfg["multistep_mode"],
            lambda_value=float(agent_cfg["lambda_value"]),
        )
        n_step = int(agent_cfg["n_step"])
        multistep_mode = str(agent_cfg["multistep_mode"])
        lambda_value = float(agent_cfg["lambda_value"])
    elif agent_kind == "sac":
        target_entropy = -action_dim if agent_cfg["target_entropy"] == "auto_negative_action_dim" else agent_cfg["target_entropy"]
        agent = SACAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            actor_hidden=list(agent_cfg["actor_hidden"]),
            critic_hidden=list(agent_cfg["critic_hidden"]),
            gamma=agent_cfg["gamma"],
            actor_lr=agent_cfg["actor_lr"],
            critic_lr=agent_cfg["critic_lr"],
            alpha_lr=agent_cfg["alpha_lr"],
            batch_size=agent_cfg["batch_size"],
            grad_clip_norm=agent_cfg["grad_clip_norm"],
            init_alpha=agent_cfg["init_alpha"],
            learn_alpha=agent_cfg["learn_alpha"],
            target_entropy=target_entropy,
            target_update=agent_cfg["target_update"],
            tau=agent_cfg["tau"],
            hard_update_interval=agent_cfg["hard_update_interval"],
            activation=agent_cfg["activation"],
            use_layernorm=agent_cfg["use_layernorm"],
            dropout=agent_cfg["dropout"],
            max_action=agent_cfg["max_action"],
            buffer_size=replay_cfg["buffer_size"],
            replay_frac_per=replay_cfg["replay_frac_per"],
            replay_frac_recent=replay_cfg["replay_frac_recent"],
            replay_recent_window=replay_cfg["replay_recent_window"],
            replay_alpha=replay_cfg["replay_alpha"],
            replay_beta_start=replay_cfg["replay_beta_start"],
            replay_beta_end=replay_cfg["replay_beta_end"],
            replay_beta_steps=replay_cfg["replay_beta_steps"],
            device=device,
            use_adamw=agent_cfg["use_adamw"],
            actor_freeze=agent_cfg["actor_freeze"],
            loss_type=agent_cfg["loss_type"],
            n_step=int(agent_cfg["n_step"]),
            multistep_mode=agent_cfg["multistep_mode"],
            lambda_value=float(agent_cfg["lambda_value"]),
        )
        n_step = int(agent_cfg["n_step"])
        multistep_mode = str(agent_cfg["multistep_mode"])
        lambda_value = float(agent_cfg["lambda_value"])
    else:
        raise ValueError("agent_kind must be 'td3' or 'sac'.")

    reid_cfg = {
        "agent_kind": agent_kind,
        "run_mode": run_mode,
        "state_mode": state_mode,
        "method_family": "reid_batch_v3_study",
        "use_v2_blend_state": True,
        "notebook_source": "RL_assisted_MPC_reid_batch_v3_ablation_unified.ipynb",
        "study_label": run_spec["study_label"],
        "ablation_group": run_spec["ablation_group"],
        "n_tests": n_tests,
        "set_points_len": set_points_len,
        "n_step": n_step,
        "multistep_mode": multistep_mode,
        "lambda_value": lambda_value,
        "warm_start": warm_start,
        "test_cycle": test_cycle,
        "predict_h": int(controller_cfg["predict_h"]),
        "cont_h": int(controller_cfg["cont_h"]),
        "use_shifted_mpc_warm_start": bool(controller_cfg["use_shifted_mpc_warm_start"]),
        "basis_family": str(run_spec.get("basis_family", reid_defaults["basis_family"])),
        "block_group_count": int(run_spec.get("block_group_count", reid_defaults["block_group_count"])),
        "block_groups": run_spec.get("block_groups", reid_defaults.get("block_groups")),
        "id_component_mode": str(run_spec.get("id_component_mode", reid_defaults.get("id_component_mode", "AB"))),
        "candidate_guard_mode": str(run_spec.get("candidate_guard_mode", reid_defaults["candidate_guard_mode"])),
        "observer_update_alignment": str(
            run_spec.get("observer_update_alignment", reid_defaults["observer_update_alignment"])
        ),
        "normalize_blend_extras": bool(
            run_spec.get("normalize_blend_extras", reid_defaults["normalize_blend_extras"])
        ),
        "blend_extra_clip": float(run_spec.get("blend_extra_clip", reid_defaults["blend_extra_clip"])),
        "blend_residual_scale": float(run_spec.get("blend_residual_scale", reid_defaults["blend_residual_scale"])),
        "log_theta_clipping": bool(run_spec.get("log_theta_clipping", reid_defaults["log_theta_clipping"])),
        "state_dim_expected": state_dim,
        "action_dim": action_dim,
        "id_solver": str(run_spec.get("id_solver", reid_defaults["id_solver"])),
        "id_window": int(run_spec.get("id_window", reid_defaults["id_window"])),
        "id_update_period": int(run_spec.get("id_update_period", reid_defaults["id_update_period"])),
        "lambda_prev": float(run_spec.get("lambda_prev", reid_defaults.get("lambda_prev", 1e-2))),
        "lambda_0": float(run_spec.get("lambda_0", reid_defaults.get("lambda_0", 1e-4))),
        "lambda_prev_A": float(run_spec.get("lambda_prev_A", reid_defaults.get("lambda_prev_A", reid_defaults.get("lambda_prev", 1e-2)))),
        "lambda_prev_B": float(run_spec.get("lambda_prev_B", reid_defaults.get("lambda_prev_B", reid_defaults.get("lambda_prev", 1e-2)))),
        "lambda_0_A": float(run_spec.get("lambda_0_A", reid_defaults.get("lambda_0_A", reid_defaults.get("lambda_0", 1e-4)))),
        "lambda_0_B": float(run_spec.get("lambda_0_B", reid_defaults.get("lambda_0_B", reid_defaults.get("lambda_0", 1e-4)))),
        "theta_low": np.asarray(run_spec.get("theta_low", reid_defaults["theta_low"]), float),
        "theta_high": np.asarray(run_spec.get("theta_high", reid_defaults["theta_high"]), float),
        "theta_low_A": float(run_spec.get("theta_low_A", reid_defaults.get("theta_low_A", -0.15))),
        "theta_high_A": float(run_spec.get("theta_high_A", reid_defaults.get("theta_high_A", 0.15))),
        "theta_low_B": float(run_spec.get("theta_low_B", reid_defaults.get("theta_low_B", -0.08))),
        "theta_high_B": float(run_spec.get("theta_high_B", reid_defaults.get("theta_high_B", 0.08))),
        "delta_A_max": float(run_spec.get("delta_A_max", reid_defaults["delta_A_max"])),
        "delta_B_max": float(run_spec.get("delta_B_max", reid_defaults["delta_B_max"])),
        "eta_smoothing_tau": float(run_spec.get("eta_smoothing_tau", reid_defaults["eta_smoothing_tau"])),
        "force_eta_constant": run_spec.get("force_eta_constant", reid_defaults["force_eta_constant"]),
        "disable_identification": bool(run_spec.get("disable_identification", reid_defaults["disable_identification"])),
        "nominal_qi": float(controller_cfg["nominal_qi"]),
        "nominal_qs": float(controller_cfg["nominal_qs"]),
        "nominal_ha": float(controller_cfg["nominal_ha"]),
        "qi_change": float(controller_cfg["qi_change"]),
        "qs_change": float(controller_cfg["qs_change"]),
        "ha_change": float(controller_cfg["ha_change"]),
        "Q1_penalty": float(controller_cfg["Q1_penalty"]),
        "Q2_penalty": float(controller_cfg["Q2_penalty"]),
        "R1_penalty": float(controller_cfg["R1_penalty"]),
        "R2_penalty": float(controller_cfg["R2_penalty"]),
        "b_min": system_data["b_min"],
        "b_max": system_data["b_max"],
        "mismatch_clip": float(controller_cfg["mismatch_clip"]),
        "innovation_scale_mode": controller_cfg["innovation_scale_mode"],
        "innovation_scale_ref": controller_cfg["innovation_scale_ref"],
        "tracking_scale_mode": controller_cfg["tracking_scale_mode"],
        "tracking_eta_tol": float(controller_cfg["tracking_eta_tol"]),
        "tracking_scale_floor": controller_cfg["tracking_scale_floor"],
        "tracking_scale_floor_mode": controller_cfg["tracking_scale_floor_mode"],
        "seed": seed,
        "regime_name": run_spec.get("regime_name"),
    }
    runtime_ctx = {
        "system": system,
        "agent": agent,
        "MPC_obj": mpc_obj,
        "steady_states": static_ctx["steady_states"],
        "min_max_dict": system_data["min_max_dict"],
        "data_min": system_data["data_min"],
        "data_max": system_data["data_max"],
        "A": A_phys,
        "B": B_phys,
        "C": C_phys,
        "A_aug": A_aug,
        "B_aug": B_aug,
        "C_aug": C_aug,
        "poles": POLYMER_OBSERVER_POLES.copy(),
        "y_sp_scenario": static_ctx["y_sp_scenario"],
        "reward_fn": static_ctx["reward_fn"],
        "system_metadata": POLYMER_SYSTEM_METADATA,
        "reward_params": static_ctx["reward_params"],
    }
    return reid_cfg, runtime_ctx


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _summary_fieldnames() -> list[str]:
    return [
        "study_label",
        "ablation_group",
        "basis_family",
        "id_component_mode",
        "force_eta_constant",
        "disable_identification",
        "observer_update_alignment",
        "normalize_blend_extras",
        "candidate_guard_mode",
        "lambda_prev_A",
        "lambda_prev_B",
        "lambda_0_A",
        "lambda_0_B",
        "theta_low_A",
        "theta_high_A",
        "theta_low_B",
        "theta_high_B",
        "regime_name",
        "final_reward",
        "best_reward",
        "tail_reward_mean",
        "tail_eta_mean",
        "tail_active_A_ratio_mean",
        "tail_active_B_ratio_mean",
        "clipping_fraction_mean",
        "fallback_fraction",
        "invalid_solve_count",
        "id_solver_failure_count",
        "update_success_count",
        "update_success_fraction",
        "update_event_count",
        "mean_abs_theta_active_tail",
        "mean_abs_theta_A_active_tail",
        "mean_abs_theta_B_active_tail",
        "run_dir",
    ]


def _build_summary_row(result_bundle: dict, tail_window: int) -> dict:
    summary = summarize_reid_run_statistics(result_bundle, tail_window=tail_window)
    lambda_prev_A = np.asarray(result_bundle.get("lambda_prev_A", []), float).reshape(-1)
    lambda_prev_B = np.asarray(result_bundle.get("lambda_prev_B", []), float).reshape(-1)
    lambda_0_A = np.asarray(result_bundle.get("lambda_0_A", []), float).reshape(-1)
    lambda_0_B = np.asarray(result_bundle.get("lambda_0_B", []), float).reshape(-1)
    theta_low_A = np.asarray(result_bundle.get("theta_low_A", []), float).reshape(-1)
    theta_high_A = np.asarray(result_bundle.get("theta_high_A", []), float).reshape(-1)
    theta_low_B = np.asarray(result_bundle.get("theta_low_B", []), float).reshape(-1)
    theta_high_B = np.asarray(result_bundle.get("theta_high_B", []), float).reshape(-1)
    return {
        "study_label": result_bundle.get("study_label"),
        "ablation_group": result_bundle.get("ablation_group"),
        "basis_family": result_bundle.get("basis_family"),
        "id_component_mode": result_bundle.get("id_component_mode"),
        "force_eta_constant": result_bundle.get("force_eta_constant"),
        "disable_identification": result_bundle.get("disable_identification"),
        "observer_update_alignment": result_bundle.get("observer_update_alignment"),
        "normalize_blend_extras": result_bundle.get("normalize_blend_extras"),
        "candidate_guard_mode": result_bundle.get("candidate_guard_mode"),
        "lambda_prev_A": float(np.mean(lambda_prev_A)) if lambda_prev_A.size > 0 else 0.0,
        "lambda_prev_B": float(np.mean(lambda_prev_B)) if lambda_prev_B.size > 0 else 0.0,
        "lambda_0_A": float(np.mean(lambda_0_A)) if lambda_0_A.size > 0 else 0.0,
        "lambda_0_B": float(np.mean(lambda_0_B)) if lambda_0_B.size > 0 else 0.0,
        "theta_low_A": float(np.mean(theta_low_A)) if theta_low_A.size > 0 else 0.0,
        "theta_high_A": float(np.mean(theta_high_A)) if theta_high_A.size > 0 else 0.0,
        "theta_low_B": float(np.mean(theta_low_B)) if theta_low_B.size > 0 else 0.0,
        "theta_high_B": float(np.mean(theta_high_B)) if theta_high_B.size > 0 else 0.0,
        "regime_name": result_bundle.get("regime_name"),
        **summary,
        "run_dir": result_bundle.get("run_dir"),
    }


def _write_study_summary_markdown(path: Path, summary_rows: list[dict]) -> None:
    ranking = sorted(summary_rows, key=lambda row: float(row["tail_reward_mean"]), reverse=True)
    top_rows = ranking[:10]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Polymer ReID Batch V3 Ablation Summary\n\n")
        handle.write(f"- Total runs: {len(summary_rows)}\n")
        if top_rows:
            handle.write(f"- Best run: `{top_rows[0]['study_label']}`\n")
            handle.write(f"- Best tail reward mean: `{top_rows[0]['tail_reward_mean']}`\n\n")
        handle.write("## Top Runs\n\n")
        for row in top_rows:
            handle.write(
                f"- `{row['study_label']}` | tier={row['ablation_group']} | basis={row['basis_family']} | "
                f"mode={row['id_component_mode']} | tail_reward_mean={row['tail_reward_mean']:.6f} | "
                f"eta_mean={row['tail_eta_mean']:.4f} | A_ratio={row['tail_active_A_ratio_mean']:.4f} | "
                f"B_ratio={row['tail_active_B_ratio_mean']:.4f}\n"
            )


def run_reid_batch_ablation_study(study_defaults: dict | None = None, repo_root: str | Path | None = None) -> dict:
    study_defaults = deepcopy(
        get_polymer_notebook_defaults("reid_batch_v3_study") if study_defaults is None else study_defaults
    )
    repo_root_path, data_dir, result_dir = _resolve_runtime_roots(study_defaults, repo_root=repo_root)
    os.chdir(repo_root_path)
    static_ctx = _build_static_polymer_context(study_defaults, repo_root_path)
    study_root = _timestamped_dir(result_dir, study_defaults["study"]["result_prefix"])
    tail_window = int(study_defaults["study"]["tail_window"])

    all_specs = []
    all_summary_rows: list[dict] = []

    tier1_specs = build_tier1_run_specs(study_defaults)
    if len(tier1_specs) != 63:
        raise ValueError(f"Tier 1 must build exactly 63 runs, got {len(tier1_specs)}.")
    all_specs.extend(tier1_specs)

    for spec in tier1_specs:
        _set_global_seed(spec.get("seed", study_defaults.get("seed", 42)))
        reid_cfg, runtime_ctx = _build_reid_cfg_and_runtime(study_defaults, static_ctx, spec)
        result_bundle = run_reid_batch_supervisor(reid_cfg=reid_cfg, runtime_ctx=runtime_ctx)
        run_dir = plot_reid_batch_theta_diagnostics(
            result_bundle=result_bundle,
            plot_cfg={
                "directory": study_root,
                "prefix_name": spec["study_label"],
                "start_episode": 1,
                "save_pdf": bool(study_defaults.get("save_pdf", False)),
                "style_profile": study_defaults.get("style_profile", "hybrid"),
            },
        )
        result_bundle["run_dir"] = str(run_dir)
        summary_row = _build_summary_row(result_bundle, tail_window=tail_window)
        all_summary_rows.append(summary_row)

    tier2_specs = build_tier2_run_specs(study_defaults, [row for row in all_summary_rows if row["ablation_group"] == "tier1"])
    all_specs.extend(tier2_specs)
    for spec in tier2_specs:
        _set_global_seed(spec.get("seed", study_defaults.get("seed", 42)))
        reid_cfg, runtime_ctx = _build_reid_cfg_and_runtime(study_defaults, static_ctx, spec)
        result_bundle = run_reid_batch_supervisor(reid_cfg=reid_cfg, runtime_ctx=runtime_ctx)
        run_dir = plot_reid_batch_theta_diagnostics(
            result_bundle=result_bundle,
            plot_cfg={
                "directory": study_root,
                "prefix_name": spec["study_label"],
                "start_episode": 1,
                "save_pdf": bool(study_defaults.get("save_pdf", False)),
                "style_profile": study_defaults.get("style_profile", "hybrid"),
            },
        )
        result_bundle["run_dir"] = str(run_dir)
        summary_row = _build_summary_row(result_bundle, tail_window=tail_window)
        all_summary_rows.append(summary_row)

    tier3_specs = build_tier3_run_specs(study_defaults, all_summary_rows)
    all_specs.extend(tier3_specs)
    for spec in tier3_specs:
        _set_global_seed(spec.get("seed", study_defaults.get("seed", 42)))
        reid_cfg, runtime_ctx = _build_reid_cfg_and_runtime(study_defaults, static_ctx, spec)
        result_bundle = run_reid_batch_supervisor(reid_cfg=reid_cfg, runtime_ctx=runtime_ctx)
        run_dir = plot_reid_batch_theta_diagnostics(
            result_bundle=result_bundle,
            plot_cfg={
                "directory": study_root,
                "prefix_name": spec["study_label"],
                "start_episode": 1,
                "save_pdf": bool(study_defaults.get("save_pdf", False)),
                "style_profile": study_defaults.get("style_profile", "hybrid"),
            },
        )
        result_bundle["run_dir"] = str(run_dir)
        summary_row = _build_summary_row(result_bundle, tail_window=tail_window)
        all_summary_rows.append(summary_row)

    summary_path = study_root / "summary.csv"
    ranking_path = study_root / "ranking.csv"
    markdown_path = study_root / "study_summary.md"
    fieldnames = _summary_fieldnames()
    _write_csv(summary_path, all_summary_rows, fieldnames)
    ranking_rows = sorted(all_summary_rows, key=lambda row: float(row["tail_reward_mean"]), reverse=True)
    _write_csv(ranking_path, ranking_rows, fieldnames)
    _write_study_summary_markdown(markdown_path, ranking_rows)
    summary_fig_dir = plot_reid_batch_ablation_summary(
        all_summary_rows,
        {
            "directory": study_root,
            "prefix_name": study_defaults["study"]["summary_prefix"],
            "save_pdf": bool(study_defaults.get("save_pdf", False)),
        },
    )
    return {
        "repo_root": str(repo_root_path),
        "data_dir": str(data_dir),
        "result_dir": str(result_dir),
        "study_root": str(study_root),
        "summary_csv": str(summary_path),
        "ranking_csv": str(ranking_path),
        "study_summary_md": str(markdown_path),
        "summary_plot_dir": str(summary_fig_dir),
        "tier1_count": len(tier1_specs),
        "tier2_count": len(tier2_specs),
        "tier3_count": len(tier3_specs),
        "total_count": len(all_summary_rows),
        "short_diagnostic_preset": dict(SHORT_DIAGNOSTIC_PRESET),
    }


if __name__ == "__main__":
    artifacts = run_reid_batch_ablation_study()
    print("Study root:", artifacts["study_root"])
    print("Summary CSV:", artifacts["summary_csv"])
    print("Ranking CSV:", artifacts["ranking_csv"])
    print("Study summary:", artifacts["study_summary_md"])
