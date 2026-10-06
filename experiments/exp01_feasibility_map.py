"""实验 1：理论边界准不准？

单向平移阶跃（z 向），名义预测前馈。在（幅值相对误差 δ，时刻误差 Δt）网格上
逐点仿真是否滑动，与 sim.theory 的解析判定对比，输出可行域图和一致率。

运行：python -m experiments.exp01_feasibility_map
"""
from functools import partial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sim.controllers import nominal_feedforward
from sim.disturbance import AXIS_Z, StepDisturbance
from sim.model import required_force
from sim.params import ActuatorParams, ObjectParams, SimParams
from sim.simulate import simulate
from sim.theory import step_feasibility

RESULTS = Path(__file__).resolve().parent.parent / "results"


def main() -> None:
    obj, act, sim = ObjectParams(), ActuatorParams(), SimParams()
    dist = StepDisturbance(amplitude=6.0, t_on=1.0, axis=AXIS_Z)
    q0 = float(required_force(np.zeros(3), obj))
    f0 = 1.1 * q0
    margin = 0.1

    amp_errs = np.linspace(-0.6, 0.6, 61)
    time_errs = np.linspace(-0.10, 0.10, 81)
    sim_ok = np.zeros((time_errs.size, amp_errs.size), dtype=bool)
    th_ok = np.zeros_like(sim_ok)

    for i, te in enumerate(time_errs):
        for j, ae in enumerate(amp_errs):
            pred = dist.with_error(ae, te)
            cmd = partial(nominal_feedforward, pred=pred, obj=obj, act=act, f0=f0, margin=margin)
            sim_ok[i, j] = not simulate(cmd, dist, obj, act, sim, f0).slipped
            th_ok[i, j] = step_feasibility(dist, pred, obj, act, f0, margin).feasible

    agreement = float((sim_ok == th_ok).mean())
    print(f"q0={q0:.3f} N, f0={f0:.3f} N, 网格 {sim_ok.shape}, 理论/仿真一致率 {agreement:.2%}")

    RESULTS.mkdir(exist_ok=True)
    np.savez(RESULTS / "exp01_feasibility_map.npz", amp_errs=amp_errs, time_errs=time_errs,
             sim_ok=sim_ok, th_ok=th_ok)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.pcolormesh(amp_errs, time_errs * 1e3, sim_ok, cmap="Greens", shading="auto", alpha=0.6)
    ax.contour(amp_errs, time_errs * 1e3, th_ok.astype(float), levels=[0.5], colors="k",
               linewidths=1.5)
    ax.set_xlabel("amplitude error  δ")
    ax.set_ylabel("timing error  Δt  [ms]  (positive = prediction late)")
    ax.set_title(f"Feasible region: simulation (green) vs theory (line), agree {agreement:.1%}")
    fig.tight_layout()
    fig.savefig(RESULTS / "exp01_feasibility_map.png", dpi=150)
    print(f"已保存 {RESULTS / 'exp01_feasibility_map.png'}")


if __name__ == "__main__":
    main()
