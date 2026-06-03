from __future__ import annotations

from DQN.supervisor_gated_dqn_agent import (
    DiscreteSupervisorGateConfig,
    DiscreteSupervisorGatedDecision,
    DiscreteSupervisorGatedMixin,
    DiscreteSupervisorPERRecentReplayBuffer,
    SOURCE_FALLBACK,
    SOURCE_HELD,
    SOURCE_NAMES,
    SOURCE_POLICY,
    SOURCE_SUPERVISOR,
    SOURCE_WARM_START,
)
from DuelingDQN.dueling_dqn_agent import DuelingDQNAgent


class SupervisorGatedDuelingDQNAgent(DiscreteSupervisorGatedMixin, DuelingDQNAgent):
    """Dueling DQN extension with the same pure-value supervisor gate as SG-DQN."""

    def __init__(self, *args, supervisor_gate_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_discrete_supervisor_gate(supervisor_gate_config)


__all__ = [
    "DiscreteSupervisorGateConfig",
    "DiscreteSupervisorGatedDecision",
    "DiscreteSupervisorPERRecentReplayBuffer",
    "SOURCE_FALLBACK",
    "SOURCE_HELD",
    "SOURCE_NAMES",
    "SOURCE_POLICY",
    "SOURCE_SUPERVISOR",
    "SOURCE_WARM_START",
    "SupervisorGatedDuelingDQNAgent",
]
