"""夹持力指令生成。每个函数返回与 t 等长的指令序列 f_cmd。

对比方法（对应研究计划第 5 步）：
- fixed_grip：固定夹持力
- reactive_feedback：纯反馈，测到加速度后才调力
- nominal_feedforward：名义预测前馈（理想预测 = 误差为 0 时的同一函数）
- TODO(第 4 步)：本文方法——随预测可信度变化的安全余量 + 滚动更新 + 不可行处理
"""
import numpy as np

from .disturbance import StepDisturbance
from .model import required_force
from .params import ActuatorParams, ObjectParams


def fixed_grip(t: np.ndarray, f_fixed: float) -> np.ndarray:
    return np.full(np.shape(t), float(f_fixed))


def reactive_feedback(t: np.ndarray, dist: StepDisturbance, obj: ObjectParams,
                      f0: float, sense_delay: float, margin: float = 0.1) -> np.ndarray:
    """延迟 sense_delay 后测得真实加速度，按 (1+margin) q 调力。"""
    q_meas = required_force(dist.accel(np.asarray(t) - sense_delay), obj)
    return np.maximum(f0, (1.0 + margin) * q_meas)


def plan_feedforward(pred: StepDisturbance, obj: ObjectParams, act: ActuatorParams,
                     f0: float, margin: float) -> tuple[float, float]:
    """单次阶跃的闭式参数：目标力 T 与指令切换时刻 t_cmd。

    T = clip(max(f0, (1+margin) q(Â)), 上限 f_max)
    t_cmd = t̂ - tau_d - (T - f0) / r   （使力恰好在预测时刻 t̂ 达到 T）
    """
    q_hat = float(required_force(pred.step_accel(), obj))
    target = min(max(f0, (1.0 + margin) * q_hat), act.f_max)
    t_cmd = pred.t_on - act.delay - (target - f0) / act.rate
    return target, t_cmd


def nominal_feedforward(t: np.ndarray, pred: StepDisturbance, obj: ObjectParams,
                        act: ActuatorParams, f0: float, margin: float = 0.1) -> np.ndarray:
    target, t_cmd = plan_feedforward(pred, obj, act, f0, margin)
    return np.where(np.asarray(t) >= t_cmd, target, f0)
