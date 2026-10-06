"""平台运动扰动与受控预测误差。

目前只有单向平移的加速度阶跃。一般波形（正弦、实测船舶运动）后续在这里补充。
"""
from dataclasses import dataclass

import numpy as np

AXIS_X, AXIS_Y, AXIS_Z = 0, 1, 2  # x 为接触法向，y/z 为切向


@dataclass(frozen=True)
class StepDisturbance:
    amplitude: float      # 加速度阶跃幅值 A [m/s^2]
    t_on: float           # 阶跃发生时刻 [s]
    axis: int = AXIS_Z

    def step_accel(self) -> np.ndarray:
        """阶跃发生后的平台加速度向量 (3,)。"""
        a = np.zeros(3)
        a[self.axis] = self.amplitude
        return a

    def accel(self, t: np.ndarray) -> np.ndarray:
        """平台加速度时间序列 (N, 3)。"""
        t = np.asarray(t, dtype=float)
        a = np.zeros((t.size, 3))
        a[t >= self.t_on, self.axis] = self.amplitude
        return a

    def with_error(self, amp_err: float, time_err: float) -> "StepDisturbance":
        """生成带受控误差的预测：幅值相对误差 delta、时刻误差 Δt（正值表示预测偏晚）。"""
        return StepDisturbance(amplitude=self.amplitude * (1.0 + amp_err),
                               t_on=self.t_on + time_err,
                               axis=self.axis)
