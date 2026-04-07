from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Sequence, Tuple


def compute_discount_power(gamma: float, n_step: int) -> float:
    gamma = float(gamma)
    n_step = int(n_step)
    if n_step < 0:
        raise ValueError("n_step must be non-negative.")
    return float(gamma ** n_step)


@dataclass
class NStepTransition:
    state: object
    action: object
    reward_n: float
    next_state_n: object
    done_n: bool
    discount_n: float
    n_actual: int


def aggregate_nstep_transition(
    transitions: Sequence[Tuple[object, object, float, object, bool]],
    gamma: float,
    max_n: int,
) -> NStepTransition:
    if not transitions:
        raise ValueError("transitions must contain at least one step.")

    max_n = int(max_n)
    if max_n <= 0:
        raise ValueError("max_n must be positive.")

    reward_n = 0.0
    n_actual = 0
    next_state_n = transitions[0][3]
    done_n = False

    for n_actual, (_, _, reward, next_state, done) in enumerate(transitions[:max_n], start=1):
        reward_n += compute_discount_power(gamma, n_actual - 1) * float(reward)
        next_state_n = next_state
        done_n = bool(done)
        if done_n:
            break

    discount_n = 0.0 if done_n else compute_discount_power(gamma, n_actual)
    state, action, _, _, _ = transitions[0]
    return NStepTransition(
        state=state,
        action=action,
        reward_n=float(reward_n),
        next_state_n=next_state_n,
        done_n=done_n,
        discount_n=float(discount_n),
        n_actual=int(n_actual),
    )


class NStepAccumulator:
    def __init__(self, gamma: float, n_step: int = 1):
        self.gamma = float(gamma)
        self.n_step = int(n_step)
        if self.n_step <= 0:
            raise ValueError("n_step must be a positive integer.")
        self._buffer: Deque[Tuple[object, object, float, object, bool]] = deque()

    def append(self, state, action, reward: float, next_state, done: bool) -> List[NStepTransition]:
        self._buffer.append((state, action, float(reward), next_state, bool(done)))
        ready: List[NStepTransition] = []

        if bool(done):
            while self._buffer:
                ready.append(aggregate_nstep_transition(tuple(self._buffer), self.gamma, self.n_step))
                self._buffer.popleft()
            return ready

        if len(self._buffer) >= self.n_step:
            ready.append(aggregate_nstep_transition(tuple(self._buffer), self.gamma, self.n_step))
            self._buffer.popleft()
        return ready

    def flush(self) -> List[NStepTransition]:
        ready: List[NStepTransition] = []
        while self._buffer:
            ready.append(aggregate_nstep_transition(tuple(self._buffer), self.gamma, self.n_step))
            self._buffer.popleft()
        return ready

    def clear(self) -> None:
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)
