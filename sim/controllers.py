"""夹持力指令生成。每个函数返回与 t 等长的指令序列 f_cmd。

对比方法（对应研究计划第 5 步）：
- fixed_grip：固定夹持力
- reactive_feedback：纯反馈，测到加速度后才调力
- nominal_feedforward：名义预测前馈（理想预测 = 误差为 0 时的同一函数）
- adaptive_feedforward：本文方法——随预测可信度变化的安全余量
- TODO(第 4 步)：滚动更新、不可行处理（目标力被 f_max 截断时的策略）
"""
import numpy as np

from .disturbance import PredictionBounds, StepDisturbance
from .model import required_force
from .params import ActuatorParams, ObjectParams

NO_BOUNDS = PredictionBounds()


def fixed_grip(t: np.ndarray, f_fixed: float) -> np.ndarray:
    return np.full(np.shape(t), float(f_fixed))


def reactive_feedback(t: np.ndarray, dist: StepDisturbance, obj: ObjectParams,
                      f0: float, sense_delay: float, margin: float = 0.1) -> np.ndarray:
    """延迟 sense_delay 后测得真实加速度，按 (1+margin) q 调力。"""
    q_meas = required_force(dist.accel(np.asarray(t) - sense_delay), obj)
    return np.maximum(f0, (1.0 + margin) * q_meas)


def worst_case_demand(pred: StepDisturbance, bounds: PredictionBounds,
                      obj: ObjectParams) -> float:
    """误差界内真实阶跃对应的最大需求 q。

    预测幅值 Â = A (1+δ)，|δ| <= δ̄ ⇒ 真实 A ∈ [Â/(1+δ̄), Â/(1-δ̄)]。
    q(a) 是 a 的仿射函数取范数，沿这条线段是凸函数，最大值在端点取得。
    """
    a_hat = pred.step_accel()
    return max(float(required_force(a_hat / (1.0 + s * bounds.amp_bound), obj))
               for s in (1.0, -1.0))


def plan_feedforward(pred: StepDisturbance, obj: ObjectParams, act: ActuatorParams,
                     f0: float, margin: float,
                     bounds: PredictionBounds = NO_BOUNDS) -> tuple[float, float]:
    """单次阶跃的闭式参数：目标力 T 与指令切换时刻 t_cmd。

    T = clip(max(f0, (1+margin) q_w), 上限 f_max)，q_w = worst_case_demand
    t_cmd = t̂ - Δt̄ - tau_d - (T - f0) / r   （使力在 t̂ - Δt̄ 达到 T）

    bounds 为零时 q_w = q(Â)、无提前量，即名义前馈。
    """
    q_w = worst_case_demand(pred, bounds, obj)
    target = min(max(f0, (1.0 + margin) * q_w), act.f_max)
    t_cmd = pred.t_on - bounds.time_bound - act.delay - (target - f0) / act.rate
    return target, t_cmd


def _step_command(t, target, t_cmd, f0):
    return np.where(np.asarray(t) >= t_cmd, target, f0)


def nominal_feedforward(t: np.ndarray, pred: StepDisturbance, obj: ObjectParams,
                        act: ActuatorParams, f0: float, margin: float = 0.1) -> np.ndarray:
    target, t_cmd = plan_feedforward(pred, obj, act, f0, margin)
    return _step_command(t, target, t_cmd, f0)


def adaptive_feedforward(t: np.ndarray, pred: StepDisturbance, bounds: PredictionBounds,
                         obj: ObjectParams, act: ActuatorParams, f0: float,
                         margin: float = 0.1) -> np.ndarray:
    """随预测可信度变化的安全余量。

    幅值余量：目标力覆盖误差界内的最坏需求 q_w，而不是 q(Â)；
    时刻余量：把力到位时刻提前 Δt̄。
    可信度越高（界越窄），余量越小；δ̄ = Δt̄ = 0 时与 nominal_feedforward 相同。
    margin 只用于覆盖误差界以外的模型误差。
    """
    target, t_cmd = plan_feedforward(pred, obj, act, f0, margin, bounds)
    return _step_command(t, target, t_cmd, f0)
