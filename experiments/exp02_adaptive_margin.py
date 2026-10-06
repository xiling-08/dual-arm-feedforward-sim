"""实验 2：随预测可信度变化的安全余量，比固定余量好在哪里？

单向平移阶跃（z 向）。预测器给出误差界 (δ̄, Δt̄) 作为可信度，真实误差在界内均匀采样。
在不同可信度等级下比较三种前馈：
- nominal：名义前馈，固定 10% 余量，不看可信度；
- fixed-conservative：按最低可信度（最宽误差界）一次定死的余量；
- adaptive：本文方法，余量随当前误差界变化。
指标：滑动率、平均峰值夹持力（调力代价）。同时检查 robust_guarantee 为真的样本
（且真实误差在界内）在仿真中是否全部不滑，以及 step_feasibility 与仿真的一致率。

运行：python -m experiments.exp02_adaptive_margin
"""
from functools import partial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sim.controllers import adaptive_feedforward
from sim.disturbance import AXIS_Z, PredictionBounds, StepDisturbance
from sim.model import required_force
from sim.params import ActuatorParams, ObjectParams, SimParams
from sim.simulate import simulate
from sim.theory import robust_guarantee, step_feasibility

RESULTS = Path(__file__).resolve().parent.parent / "results"

AMP_BOUNDS = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5])  # 幅值误差界 δ̄
TIME_PER_AMP = 0.1        # Δt̄ = 0.1 s × δ̄，两类误差界同步放宽
N_SAMPLES = 200           # 每个可信度等级的蒙特卡洛样本数
MARGIN = 0.1
SEED = 0

METHODS = ["nominal", "fixed-conservative", "adaptive"]
COLORS = {"nominal": "#2a78d6", "fixed-conservative": "#eb6834", "adaptive": "#1baf7a"}
MARKERS = {"nominal": "o", "fixed-conservative": "s", "adaptive": "^"}
MARKER_SIZE = {"nominal": 6, "fixed-conservative": 10, "adaptive": 6}  # 曲线重合时仍能看见底下那条


def level_bounds(amp_bound: float) -> PredictionBounds:
    return PredictionBounds(amp_bound=amp_bound, time_bound=TIME_PER_AMP * amp_bound)


def main() -> None:
    obj, act, sim = ObjectParams(), ActuatorParams(), SimParams()
    dist = StepDisturbance(amplitude=6.0, t_on=1.0, axis=AXIS_Z)
    f0 = 1.1 * float(required_force(np.zeros(3), obj))
    rng = np.random.default_rng(SEED)
    widest = level_bounds(AMP_BOUNDS.max())
    method_bounds = {"nominal": lambda b: PredictionBounds(),
                     "fixed-conservative": lambda b: widest,
                     "adaptive": lambda b: b}

    slip_rate = {m: np.zeros(AMP_BOUNDS.size) for m in METHODS}
    peak_force = {m: np.zeros(AMP_BOUNDS.size) for m in METHODS}
    n_agree = n_total = n_guaranteed = n_guarantee_violated = 0

    for i, ab in enumerate(AMP_BOUNDS):
        bounds = level_bounds(ab)
        amp_errs = rng.uniform(-bounds.amp_bound, bounds.amp_bound, N_SAMPLES)
        time_errs = rng.uniform(-bounds.time_bound, bounds.time_bound, N_SAMPLES)
        for m in METHODS:
            b = method_bounds[m](bounds)
            slips, peaks = [], []
            for ae, te in zip(amp_errs, time_errs):
                pred = dist.with_error(ae, te)
                cmd = partial(adaptive_feedforward, pred=pred, bounds=b, obj=obj, act=act,
                              f0=f0, margin=MARGIN)
                res = simulate(cmd, dist, obj, act, sim, f0)
                slips.append(res.slipped)
                peaks.append(res.f.max())
                th = step_feasibility(dist, pred, obj, act, f0, MARGIN, b).feasible
                n_agree += th == (not res.slipped)
                n_total += 1
                # 保证只对落在该方法所用误差界内的真实误差成立
                if b.contains(ae, te) and robust_guarantee(pred, b, obj, act, f0, MARGIN):
                    n_guaranteed += 1
                    n_guarantee_violated += res.slipped
            slip_rate[m][i] = np.mean(slips)
            peak_force[m][i] = np.mean(peaks)

    q1 = float(required_force(dist.step_accel(), obj))
    print(f"f0={f0:.3f} N, q1={q1:.3f} N, 每级 {N_SAMPLES} 样本")
    print(f"{'δ̄':>5} {'Δt̄[ms]':>8} | " + " | ".join(f"{m:>18}" for m in METHODS))
    for i, ab in enumerate(AMP_BOUNDS):
        cells = [f"slip {slip_rate[m][i]:5.1%} F {peak_force[m][i]:5.2f}" for m in METHODS]
        print(f"{ab:5.2f} {TIME_PER_AMP * ab * 1e3:8.1f} | " + " | ".join(cells))
    print(f"理论/仿真一致率 {n_agree / n_total:.2%}（{n_total} 次仿真）")
    print(f"有鲁棒保证的样本 {n_guaranteed} 个，其中仿真滑动 {n_guarantee_violated} 个")

    RESULTS.mkdir(exist_ok=True)
    np.savez(RESULTS / "exp02_adaptive_margin.npz", amp_bounds=AMP_BOUNDS,
             time_bounds=TIME_PER_AMP * AMP_BOUNDS,
             **{f"slip_{m}": slip_rate[m] for m in METHODS},
             **{f"peak_{m}": peak_force[m] for m in METHODS})

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for m in METHODS:
        style = dict(color=COLORS[m], marker=MARKERS[m], markersize=MARKER_SIZE[m], linewidth=2, label=m)
        axes[0].plot(AMP_BOUNDS, slip_rate[m] * 100, **style)
        axes[1].plot(AMP_BOUNDS, peak_force[m], **style)
    axes[0].set_ylabel("slip rate  [%]")
    axes[1].set_ylabel("mean peak grip force  [N]")
    axes[0].set_title("Slip rate vs prediction uncertainty")
    axes[1].set_title("Force cost vs prediction uncertainty")
    for ax in axes:
        ax.set_xlabel(rf"amplitude error bound $\bar{{\delta}}$   (timing bound $\bar{{\Delta t}}$ = {TIME_PER_AMP:g} s $\times\ \bar{{\delta}}$)")
        ax.grid(True, color="0.9", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(RESULTS / "exp02_adaptive_margin.png", dpi=150)
    print(f"已保存 {RESULTS / 'exp02_adaptive_margin.png'}")


if __name__ == "__main__":
    main()
