"""随预测可信度变化的安全余量（adaptive_feedforward）的测试。"""
from functools import partial

import numpy as np
import pytest

from sim.controllers import adaptive_feedforward, nominal_feedforward, worst_case_demand
from sim.disturbance import AXIS_Z, PredictionBounds, StepDisturbance
from sim.model import required_force
from sim.params import ActuatorParams, ObjectParams, SimParams
from sim.simulate import simulate
from sim.theory import robust_guarantee, step_feasibility

OBJ, ACT, SIM = ObjectParams(), ActuatorParams(), SimParams()
DIST = StepDisturbance(amplitude=6.0, t_on=1.0, axis=AXIS_Z)
F0 = 1.1 * float(required_force(np.zeros(3), OBJ))
BOUNDS = PredictionBounds(amp_bound=0.3, time_bound=0.04)


def _slipped(cmd) -> bool:
    return simulate(cmd, DIST, OBJ, ACT, SIM, F0).slipped


def _adaptive(pred, bounds=BOUNDS, act=ACT):
    return partial(adaptive_feedforward, pred=pred, bounds=bounds, obj=OBJ, act=act,
                   f0=F0, margin=0.1)


def test_bounds_validation():
    with pytest.raises(ValueError):
        PredictionBounds(amp_bound=1.0)
    with pytest.raises(ValueError):
        PredictionBounds(time_bound=-0.01)


def test_zero_bounds_reduces_to_nominal():
    pred = DIST.with_error(0.2, 0.01)
    t = np.arange(0.0, SIM.t_end, SIM.dt)
    nominal = nominal_feedforward(t, pred, OBJ, ACT, F0, margin=0.1)
    np.testing.assert_array_equal(_adaptive(pred, PredictionBounds())(t), nominal)


def test_worst_case_demand_covers_true_demand():
    for amp_err in np.linspace(-BOUNDS.amp_bound, BOUNDS.amp_bound, 7):
        pred = DIST.with_error(amp_err, 0.0)
        q1 = float(required_force(DIST.step_accel(), OBJ))
        assert worst_case_demand(pred, BOUNDS, OBJ) >= q1 - 1e-12


def test_margin_shrinks_with_credibility():
    """可信度越高（界越窄），目标力越低、加力越晚。"""
    pred = DIST.with_error(0.0, 0.0)
    t = np.arange(0.0, SIM.t_end, SIM.dt)
    wide = _adaptive(pred, PredictionBounds(0.4, 0.06))(t)
    narrow = _adaptive(pred, PredictionBounds(0.1, 0.01))(t)
    assert narrow.max() < wide.max()
    assert np.argmax(narrow > F0) > np.argmax(wide > F0)


@pytest.mark.parametrize("amp_err, time_err", [(-0.25, 0.035), (0.25, -0.04), (-0.25, -0.04)])
def test_guaranteed_inside_bounds(amp_err, time_err):
    pred = DIST.with_error(amp_err, time_err)
    assert robust_guarantee(pred, BOUNDS, OBJ, ACT, F0, 0.1)
    assert not _slipped(_adaptive(pred))


def test_adaptive_survives_where_nominal_slips():
    pred = DIST.with_error(-0.25, 0.035)
    nominal = partial(nominal_feedforward, pred=pred, obj=OBJ, act=ACT, f0=F0, margin=0.1)
    assert _slipped(nominal)
    assert not _slipped(_adaptive(pred))


@pytest.mark.parametrize("amp_err, time_err, expected", [
    (-0.25, 0.035, True),   # 界内
    (-0.6, 0.0, False),     # 幅值低估远超界：违反条件 1
    (0.0, 0.12, False),     # 偏晚 120 ms，远超 Δt̄ + (T-q1)/r：违反条件 2
    (0.5, -0.2, True),      # 幅值超界但为高估、偏早：仍可行
])
def test_theory_matches_simulation_away_from_boundary(amp_err, time_err, expected):
    pred = DIST.with_error(amp_err, time_err)
    th = step_feasibility(DIST, pred, OBJ, ACT, F0, 0.1, BOUNDS).feasible
    assert (not _slipped(_adaptive(pred))) == th == expected


def test_no_guarantee_when_target_clipped():
    weak = ActuatorParams(f_max=8.0)
    pred = DIST.with_error(-0.25, 0.0)
    assert not robust_guarantee(pred, BOUNDS, OBJ, weak, F0, 0.1)
