"""Change acceptance only; native proposal, readiness and replay remain intact."""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import replace
import numpy as np


def normalized_error(y, reference, reward_params):
    y, reference = np.asarray(y, float), np.asarray(reference, float)
    bands = np.maximum(np.asarray(reward_params["k_rel"])*np.abs(reference),
                       np.asarray(reward_params["band_floor_phys"]))
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(reference)) or np.any(bands <= 0):
        return float("inf")
    return float(np.max(np.abs(y-reference)/bands))


def calibrate(y_before, references, reward_params):
    errors = np.array([normalized_error(y,r,reward_params) for y,r in zip(y_before,references)])
    if not len(errors) or not np.all(np.isfinite(errors)):
        raise ValueError("Threshold calibration requires finite nominal warm-up measurements.")
    return max(1.,1.25*float(np.quantile(errors,.95)))


class ThresholdSession:
    """Suspend the unchanged runner only to expose current physical measurements."""
    def __init__(self, compiled, context, *, threshold=None):
        self.compiled, self.context, self.threshold = compiled, context, threshold
        self.records = []
        self.agents = {}

    def wants(self, key, environment):
        return self.threshold is not None

    def is_active(self, key):
        return self.threshold is not None

    def before(self, site, environment):
        from utils.helpers import reverse_min_max
        step = int(environment[self.compiled.step_name])
        reference = reverse_min_max(environment["y_sp"][step]+environment["y_ss_scaled"],
                                    environment["data_min"][2:],environment["data_max"][2:])
        self.context["error"] = normalized_error(environment["system"].current_output,reference,
                                                environment["reward_params"])
        self.context["threshold"] = self.threshold
        self.context["calls"] = 0
        frozen = (int(environment["horizon_cfg"].get("post_warm_start_action_freeze_subepisodes",0))
                  if site.channel == "horizon" else int(environment.get("td3_phase1_action_freeze_subepisodes",0)))
        self.context["eligible"] = step > int(environment["warm_start_step"]) + frozen*int(environment["time_in_sub_episodes"])

    def after(self, site, environment):
        decision = environment[site.result_name]
        self.records.append((int(environment[self.compiled.step_name]), site.channel, self.context["error"],
                             int(getattr(decision,"selected_source",-1)),
                             bool(getattr(decision,"decision_taken",True)), int(self.context["calls"])))

    def finish(self, environment):
        self.agents = {name: environment[name] for name in
                       ("horizon_agent","weight_agent","markov_agent","residual_agent")
                       if environment.get(name) is not None}

    def detach(self):
        pass


@contextmanager
def acceptance_override(context):
    """Process-local override: TD3 chooses before logging; DQN fixes its trace.

    Scores are retained as shadow diagnostics. They have no effect on threshold
    acceptance. This implementation does not claim critic inference cost savings.
    """
    import TD3Agent.supervisor_gated_agent as td3
    from DQN.supervisor_gated_dqn_agent import SupervisorGatedDQNAgent
    original_choice = td3.choose_policy_or_supervisor
    original_discrete = SupervisorGatedDQNAgent.select_action_with_supervisor
    def choose(*args, **kwargs):
        context["calls"] += 1
        if not context.get("eligible",True):
            return original_choice(*args,**kwargs)
        return "policy" if context["error"] <= context["threshold"] else "supervisor"
    def discrete(self, *args, **kwargs):
        decision = original_discrete(self,*args,**kwargs)
        context["calls"] += 1
        if not context.get("eligible",True):
            return decision
        eligible = self.train_steps >= int(self.supervisor_gate_config.min_train_steps_before_policy_gate)
        policy = bool(eligible and context["error"] <= context["threshold"])
        selected_source = 2 if policy else 1
        action = decision.policy_action if policy else decision.supervisor_action
        self.sg_selected_source_trace[-1] = float(selected_source)
        return replace(decision,action=int(action),selected_source=selected_source)
    td3.choose_policy_or_supervisor = choose
    SupervisorGatedDQNAgent.select_action_with_supervisor = discrete
    try:
        yield
    finally:
        td3.choose_policy_or_supervisor = original_choice
        SupervisorGatedDQNAgent.select_action_with_supervisor = original_discrete
