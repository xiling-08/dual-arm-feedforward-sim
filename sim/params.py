"""仿真参数。所有物理量使用 SI 单位。数值是占位值，实机核实后再替换。"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ObjectParams:
    m: float = 0.5      # 物体质量 [kg]
    mu: float = 0.5     # 接触摩擦系数
    g: float = 9.81     # 重力加速度 [m/s^2]


@dataclass(frozen=True)
class ActuatorParams:
    """调力响应模型：纯延迟 + 斜率限制 + 上下限。"""
    delay: float = 0.05   # 纯延迟 tau_d [s]
    rate: float = 40.0    # 最大调力速率 r [N/s]
    f_min: float = 0.0    # 夹持力下限 [N]
    f_max: float = 30.0   # 夹持力上限 [N]


@dataclass(frozen=True)
class SimParams:
    dt: float = 1e-3      # 仿真步长 [s]
    t_end: float = 2.0    # 仿真时长 [s]
