"""单次仿真：扰动 -> 需求 q(t)，指令 -> 实际力 f(t)，比较得到防滑裕度。"""
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .actuator import simulate_actuator
from .disturbance import StepDisturbance
from .model import required_force
from .params import ActuatorParams, ObjectParams, SimParams

SLIP_TOL = 1e-9


@dataclass
class SimResult:
    t: np.ndarray
    a: np.ndarray
    q_req: np.ndarray
    f_cmd: np.ndarray
    f: np.ndarray

    @property
    def margin(self) -> np.ndarray:
        """防滑裕度 f - q，负值表示滑动。"""
        return self.f - self.q_req

    @property
    def min_margin(self) -> float:
        return float(self.margin.min())

    @property
    def slipped(self) -> bool:
        return self.min_margin < -SLIP_TOL


def simulate(command: Callable[[np.ndarray], np.ndarray], dist: StepDisturbance,
             obj: ObjectParams, act: ActuatorParams, sim: SimParams, f0: float) -> SimResult:
    t = np.arange(0.0, sim.t_end, sim.dt)
    a = dist.accel(t)
    q_req = required_force(a, obj)
    f_cmd = command(t)
    f = simulate_actuator(f_cmd, sim.dt, act, f_init=f0)
    return SimResult(t=t, a=a, q_req=q_req, f_cmd=f_cmd, f=f)
