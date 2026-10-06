"""单次阶跃 + 前馈（名义 / 可信度自适应余量）下的解析可行性判定。

注意：这是按本仓库模型推出的示例版本，需要与论文第 IV 节的“双条件定理”逐项对齐
（符号、假设、是否含转矩项）。仿真实验 exp01 用它画“理论边界”。

设 q0 = q(静止)，q1 = q(真实阶跃后)，T 为前馈目标力，Δt = t̂ - t_on（预测偏晚为正），
Δt̄ 为计划的时刻提前量（名义前馈为 0）。
前馈下实际力在 [t̂ - Δt̄ - (T-f0)/r, t̂ - Δt̄] 内从 f0 线性升到 T，之后保持 T。
由于阶跃后需求恒为 q1、实际力单调不减，可行 <=> 静止时不滑 且 f(t_on) >= q1：

    条件 0（静止）：f0 >= q0
    条件 1（幅值）：T >= q1
    条件 2（时刻）：f0 >= q1  或  max(Δt - Δt̄, 0) <= (T - q1) / r

假设：指令切换时刻 t_cmd >= 0（预测提前量足够）。

鲁棒保证（robust_guarantee，不需要知道真实扰动）：若 f0 >= q0、T 未被 f_max 截断
（T >= q_w，q_w 为误差界内最坏需求）且 t_cmd >= 0，则对界内任意误差
|δ| <= δ̄、Δt <= Δt̄ 都有 T >= q_w >= q1 且 Δt - Δt̄ <= 0，条件 0–2 全部成立。
这只是开环、单次阶跃下的结论，不涉及闭环稳定性。
"""
from dataclasses import dataclass

import numpy as np

from .controllers import NO_BOUNDS, plan_feedforward, worst_case_demand
from .disturbance import PredictionBounds, StepDisturbance
from .model import required_force
from .params import ActuatorParams, ObjectParams


@dataclass(frozen=True)
class StepFeasibility:
    rest_ok: bool
    amp_ok: bool
    timing_ok: bool
    target: float
    q1: float

    @property
    def feasible(self) -> bool:
        return self.rest_ok and self.amp_ok and self.timing_ok


def step_feasibility(dist: StepDisturbance, pred: StepDisturbance, obj: ObjectParams,
                     act: ActuatorParams, f0: float, margin: float,
                     bounds: PredictionBounds = NO_BOUNDS) -> StepFeasibility:
    q0 = float(required_force(np.zeros(3), obj))
    q1 = float(required_force(dist.step_accel(), obj))
    target, _ = plan_feedforward(pred, obj, act, f0, margin, bounds)
    late = max(pred.t_on - dist.t_on - bounds.time_bound, 0.0)
    return StepFeasibility(
        rest_ok=f0 >= q0,
        amp_ok=target >= q1,
        timing_ok=(f0 >= q1) or (late <= (target - q1) / act.rate),
        target=target,
        q1=q1,
    )


def robust_guarantee(pred: StepDisturbance, bounds: PredictionBounds, obj: ObjectParams,
                     act: ActuatorParams, f0: float, margin: float) -> bool:
    """误差界内任意真实扰动都不滑（见模块文档）。False 不代表一定滑，只是无保证。"""
    q0 = float(required_force(np.zeros(3), obj))
    target, t_cmd = plan_feedforward(pred, obj, act, f0, margin, bounds)
    return f0 >= q0 and target >= worst_case_demand(pred, bounds, obj) and t_cmd >= 0.0
