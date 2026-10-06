from functools import partial

import numpy as np
import pytest

from sim.actuator import simulate_actuator
from sim.controllers import nominal_feedforward
from sim.disturbance import AXIS_X, AXIS_Z, StepDisturbance
from sim.model import required_force
from sim.params import ActuatorParams, ObjectParams, SimParams
from sim.simulate import simulate
from sim.theory import step_feasibility

OBJ, ACT, SIM = ObjectParams(), ActuatorParams(), SimParams()


def test_required_force_at_rest():
    q0 = required_force(np.zeros(3), OBJ)
    assert q0 == pytest.approx(OBJ.m * OBJ.g / (2 * OBJ.mu))


def test_required_force_normal_axis_adds_half():
    a = np.zeros(3)
    a[AXIS_X] = 4.0
    extra = required_force(a, OBJ) - required_force(np.zeros(3), OBJ)
    assert extra == pytest.approx(OBJ.m * 4.0 / 2)


def test_moment_not_silently_ignored():
    with pytest.raises(NotImplementedError):
        required_force(np.zeros(3), OBJ, moment=np.array([0.0, 0.0, 0.1]))


def test_actuator_delay_and_rate():
    f_cmd = np.full(400, 10.0)
    f = simulate_actuator(f_cmd, SIM.dt, ACT, f_init=0.0)
    n_delay = int(round(ACT.delay / SIM.dt))
    assert f[n_delay - 1] == 0.0
    assert f[n_delay + 99] == pytest.approx(ACT.rate * SIM.dt * 100)


@pytest.mark.parametrize("amp_err, time_err, expected", [
    (0.0, 0.0, True),     # 无误差：可行
    (-0.5, 0.0, False),   # 幅值严重低估：违反条件 1
    (0.0, 0.05, False),   # 预测偏晚 50 ms：违反条件 2
    (0.0, -0.05, True),   # 预测偏早：只是提前加力
])
def test_theory_matches_simulation_away_from_boundary(amp_err, time_err, expected):
    dist = StepDisturbance(amplitude=6.0, t_on=1.0, axis=AXIS_Z)
    f0 = 1.1 * float(required_force(np.zeros(3), OBJ))
    pred = dist.with_error(amp_err, time_err)
    cmd = partial(nominal_feedforward, pred=pred, obj=OBJ, act=ACT, f0=f0, margin=0.1)
    sim_feasible = not simulate(cmd, dist, OBJ, ACT, SIM, f0).slipped
    th_feasible = step_feasibility(dist, pred, OBJ, ACT, f0, 0.1).feasible
    assert sim_feasible == th_feasible == expected
