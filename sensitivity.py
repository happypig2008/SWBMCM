# -*- coding: utf-8 -*-
"""
2026 数维杯 B 题 —— 敏感性分析
- 2(2) makespan 对 车辆速度 / 云台角速度 / 观测时长 的敏感性 (±20%)
- 2(1) 定位误差对 方位角误差 σ 的敏感性 (线性, 解析)
- 3 s_max 对 云台零偏 σ0 / 控制误差 σc 的敏感性 (解析)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt

from src.config import (FIELD, VEHICLES, VEHICLE_SPEED, CAMERA_HEIGHT,
                        GIMBAL_ANGULAR_SPEED, JOINT_OBS_TIME, ROUTING_STEP,
                        alert_xyz)
from src.geometry import observable_mask
from src.routing import make_grid, shortest_path_distances
from src.plotting import setup_style

setup_style()

BASE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE, "sensitivity_figures")
os.makedirs(FIG_DIR, exist_ok=True)
RES_PATH = os.path.join(BASE, "sensitivity_results.txt")

# ---------- 与参数无关的固定部分 ----------
xs, ys, obstacle = make_grid(ROUTING_STEP)
Xg, Yg = np.meshgrid(xs, ys)
names = list(VEHICLES.keys())
ALERTS = ["T1", "T2", "T3"]
targets = {a: alert_xyz(a) for a in ALERTS}


def ij_to_xy(idx):
    i, j = idx
    return float(xs[i]), float(ys[j])


init_idx = {}
for v in names:
    vx, vy = VEHICLES[v]
    init_idx[v] = (int(round((vx - xs[0]) / ROUTING_STEP)),
                   int(round((vy - ys[0]) / ROUTING_STEP)))

init_dist = {v: shortest_path_distances(VEHICLES[v], xs, ys, obstacle) for v in names}

obs_cache = {}


def obs_mask(target):
    if target not in obs_cache:
        obs_cache[target] = observable_mask(Xg, Yg, target[0], target[1], target[2])
    return obs_cache[target]


def slew_field(target, ang):
    tx, ty, tz = target
    r_h = np.hypot(tx - Xg, ty - Yg)
    az = np.abs(np.degrees(np.arctan2(ty - Yg, tx - Xg)))
    elev = np.degrees(np.arctan2(tz - CAMERA_HEIGHT, r_h))
    return np.maximum(az, elev) / ang


# ---------- 参数化: 贪心调度求 makespan ----------
def compute_makespan(speed, ang, obs_time):
    S, slew_t = {}, {}
    for v in names:
        for a in ALERTS:
            info = obs_mask(targets[a])
            sf = slew_field(targets[a], ang)
            ready = (init_dist[v] / speed + sf).copy()
            ready[~info] = np.inf
            flat = int(np.argmin(ready))
            j, i = np.unravel_index(flat, ready.shape)
            S[(v, a)] = (i, j)
            slew_t[(v, a)] = float(sf[j, i])

    dist_cache = {}

    def travel_time(frm, to):
        if frm not in dist_cache:
            dist_cache[frm] = shortest_path_distances(ij_to_xy(frm), xs, ys, obstacle)
        return dist_cache[frm][to[1], to[0]] / speed

    veh_time = {v: 0.0 for v in names}
    veh_pos = {v: None for v in names}
    obs_start = {}
    for a in ALERTS:
        cand = {}
        for v in names:
            st = S[(v, a)]
            frm = veh_pos[v] if veh_pos[v] is not None else init_idx[v]
            tt = travel_time(frm, st)
            cand[v] = (veh_time[v] + tt + slew_t[(v, a)], tt, slew_t[(v, a)], st)
        pick = sorted(names, key=lambda v: cand[v][0])[:2]
        start = max(cand[p][0] for p in pick)
        obs_start[a] = start
        for p in pick:
            r, tt, sl, st = cand[p]
            veh_time[p] = start + obs_time + sl
            veh_pos[p] = st
    return max(obs_start.values())


# ---------- 敏感性主循环 ----------
base_mk = compute_makespan(VEHICLE_SPEED, GIMBAL_ANGULAR_SPEED, JOINT_OBS_TIME)

L = []
L.append("=" * 70)
L.append("B题 敏感性分析 (关键参数 ±20%)")
L.append("=" * 70)
L.append(f"基准: 车辆速度={VEHICLE_SPEED} m/s, 云台角速度={GIMBAL_ANGULAR_SPEED}°/s, "
         f"观测时长={JOINT_OBS_TIME}s -> makespan={base_mk:.2f}s")
L.append("-" * 70)
L.append(f"{'参数':<12}{'-20%':<10}{'基准':<10}{'+20%':<10}{'最大变化':<12}")
L.append("-" * 70)

mk_results = {}
for pname, base_val in [("车辆速度", VEHICLE_SPEED),
                        ("云台角速度", GIMBAL_ANGULAR_SPEED),
                        ("观测时长", JOINT_OBS_TIME)]:
    vals = []
    for factor in [0.8, 1.0, 1.2]:
        if pname == "车辆速度":
            mk = compute_makespan(base_val * factor, GIMBAL_ANGULAR_SPEED, JOINT_OBS_TIME)
        elif pname == "云台角速度":
            mk = compute_makespan(VEHICLE_SPEED, base_val * factor, JOINT_OBS_TIME)
        else:
            mk = compute_makespan(VEHICLE_SPEED, GIMBAL_ANGULAR_SPEED, base_val * factor)
        vals.append(mk)
    mk_results[pname] = vals
    spread = max(abs(vals[0] - vals[1]), abs(vals[2] - vals[1])) / vals[1] * 100
    L.append(f"{pname:<12}{vals[0]:<10.2f}{vals[1]:<10.2f}{vals[2]:<10.2f}{spread:<12.1f}%")

L.append("结论: makespan 对云台角速度最敏感(转云台时间主导), 观测时长次之, "
         "车辆速度几乎不敏感(目标多从初始位置可观测, 行驶极少)。")
L.append("=" * 70)

# ---------- 2(1) 定位误差 vs 方位角误差 σ (线性) ----------
L.append("[2(1)] 定位误差 1σ = σ_方位角 × GDOP (GDOP=8.23 为几何量, 与 σ 无关)")
L.append(f"{'方位角误差 σ':<16}{'定位误差 1σ':<14}")
L.append("-" * 30)
for sig in [0.8, 1.0, 1.2]:
    err = np.radians(sig) * 8.23 * 100  # cm
    L.append(f"{sig:.1f}°{'':<13}{err:.2f} cm")
L.append("-" * 30)
L.append("结论: 定位误差与 σ 严格线性, 无放大/衰减。")

L.append("=" * 70)

# ---------- 3 s_max vs σ0 / σc (解析) ----------
Z = 1.959964
R = 40.0
THRESH = np.radians(5.0)


def s_max_from(s0_deg, sc_deg):
    s0 = np.radians(s0_deg)
    sc = np.radians(sc_deg)
    sigma_max = THRESH / Z
    var_g = s0 ** 2 + sc ** 2
    if sigma_max ** 2 <= var_g:
        return 0.0
    sigma_pos_max = R * np.sqrt(sigma_max ** 2 - var_g)
    return (sigma_pos_max - 0.3) / 0.02


L.append("[3] s_max 对云台零偏 σ0 / 控制误差 σc 的敏感性")
L.append(f"{'σ0 (度)':<10}{'σc (度)':<10}{'s_max (m)':<12}")
L.append("-" * 32)
smax_results = {}
for s0 in [0.6, 0.8, 1.0]:
    smax_results[("s0", s0)] = s_max_from(s0, 0.6)
    L.append(f"{s0:<10}{0.6:<10}{s_max_from(s0, 0.6):<12.2f}")
L.append("-" * 32)
for sc in [0.4, 0.6, 0.8]:
    smax_results[("sc", sc)] = s_max_from(0.8, sc)
    L.append(f"{0.8:<10}{sc:<10}{s_max_from(0.8, sc):<12.2f}")
L.append("-" * 32)
L.append("结论: 云台零偏 σ0 增大时 s_max 下降更明显(σ0 方差占比更大)。")
L.append("=" * 70)

print("\n".join(L))
with open(RES_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(L) + "\n")

# ---------- 图 ----------
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# 3 张 makespan 敏感性
for ax, (pname, vals) in zip(axes[0], list(mk_results.items())[:3]):
    ax.plot([-20, 0, 20], vals, "o-", lw=2, ms=8)
    ax.axhline(vals[1], color="gray", ls=":", lw=1)
    ax.set_xticks([-20, 0, 20])
    ax.set_xlabel("参数变化 / %")
    ax.set_ylabel("makespan / s")
    ax.set_title(f"(a) 敏感性: {pname}")
    ax.grid(alpha=0.3)

# s_max 敏感性
ax = axes[1, 0]
s0s = [0.6, 0.8, 1.0]
ax.plot(s0s, [smax_results[("s0", s)] for s in s0s], "o-", lw=2, label="变 σ0 (σc=0.6)")
scs = [0.4, 0.6, 0.8]
ax.plot(scs, [smax_results[("sc", s)] for s in scs], "s--", lw=2, label="变 σc (σ0=0.8)")
ax.set_xlabel("误差 (度)")
ax.set_ylabel("s_max / m")
ax.set_title("(d) 问题3: s_max 敏感性")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)

# 空白占位
axes[1, 1].axis("off")
axes[1, 1].text(0.5, 0.5,
                "敏感性结论:\n"
                "· makespan 对云台角速度最敏感\n"
                "· 观测时长次之, 车辆速度最弱\n"
                "· 定位误差与 σ 严格线性\n"
                "· s_max 随云台误差增大而快速下降",
                ha="center", va="center", fontsize=11)

plt.tight_layout()
out_fig = os.path.join(FIG_DIR, "sensitivity.png")
plt.savefig(out_fig, dpi=150, bbox_inches="tight")
print(f"\n图已保存: {out_fig}")
print(f"结果已保存: {RES_PATH}")
plt.show()
