"""调力响应：纯延迟 + 斜率限制 + 上下限饱和。"""
import numpy as np

from .params import ActuatorParams


def simulate_actuator(f_cmd: np.ndarray, dt: float, act: ActuatorParams,
                      f_init: float) -> np.ndarray:
    """给定指令序列 f_cmd，返回实际夹持力序列 f。

    延迟期间执行器看到的指令视为 f_init。
    """
    n_delay = int(round(act.delay / dt))
    step = act.rate * dt
    f = np.empty_like(f_cmd, dtype=float)
    cur = float(f_init)
    for k in range(len(f_cmd)):
        target = f_cmd[k - n_delay] if k >= n_delay else f_init
        target = min(max(target, act.f_min), act.f_max)
        cur += min(max(target - cur, -step), step)
        f[k] = cur
    return f
