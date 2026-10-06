"""需求映射：平台加速度 -> 防滑所需的最小内部力 q。

几何假设（两点夹持）：
- 物体由两侧夹爪在 body 系 x 轴两端夹持，接触法向为 ±x；
- 平台只做平移，物体与平台无相对运动；
- 切向载荷由两个接触点平分。

记物体所需合外力（由接触提供）F = m (a - g_vec)，g_vec = (0, 0, -g)。
两个接触点法向力 n1 = f + Fx/2, n2 = f - Fx/2，其中 f 为内部力（挤压力）。
每个接触点的切向力为 |Ft|/2，库仑摩擦要求 |Ft|/2 <= mu * min(n1, n2)，
得到最小内部力：

    q(F) = |Ft| / (2 mu) + |Fx| / 2

TODO(论文 II 节)：平台转动时需加入转矩项 M（扭转摩擦、切向力不再平分），
目前传入非零 M 会直接报错，避免静默地得到错误结果。
"""
import numpy as np

from .params import ObjectParams


def contact_load(a, obj: ObjectParams) -> np.ndarray:
    """接触需要提供的合力 F，形状与 a 相同 (..., 3)。"""
    a = np.asarray(a, dtype=float)
    g_vec = np.array([0.0, 0.0, -obj.g])
    return obj.m * (a - g_vec)


def required_force(a, obj: ObjectParams, moment=None) -> np.ndarray:
    """最小内部力 q，a 形状 (..., 3)，返回形状 (...)。"""
    if moment is not None and np.any(np.asarray(moment) != 0):
        raise NotImplementedError("转矩项 M 尚未建模，见模块文档 TODO")
    F = contact_load(a, obj)
    f_normal = np.abs(F[..., 0])
    f_tangent = np.linalg.norm(F[..., 1:], axis=-1)
    return f_tangent / (2.0 * obj.mu) + f_normal / 2.0
