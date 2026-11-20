# -*- coding: utf-8 -*-
"""
2026 数维杯 B 题 —— 问题 2(2) (重写版)
三个警报并发下的车辆复用与调度 (最小化最大观测开始时间 makespan)
- 绕楼最短路径 (Dijkstra, 按需缓存)
- 固定站模型: 每(车,警报)取"从初始位置最优"的站 S[(v,a)]
- baseline: 最近指派启发式 (正确计入车辆占用时间)
- 优化: 枚举全部 1000 种指派 × 车辆任务顺序, 事件驱动模拟求 makespan
"""
import os
import sys
from itertools import combinations, permutations, product

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from src.config import (FIELD, VEHICLES, VEHICLE_SPEED, CAMERA_HEIGHT,
                        GIMBAL_ANGULAR_SPEED, JOINT_OBS_TIME, ROUTING_STEP,
                        alert_xyz)
from src.geometry import observable_mask
from src.routing import make_grid, shortest_path_distances
from src.plotting import setup_style

setup_style()

SPEED = VEHICLE_SPEED            # 3 m/s
ANG = GIMBAL_ANGULAR_SPEED       # 30 °/s
OBS = JOINT_OBS_TIME             # 5 s
ALERTS = ["T1", "T2", "T3"]

BASE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
RES_PATH = os.path.join(BASE, "results_2_2.txt")

# ---------- 网格 / 警报 / 距离场 ----------
xs, ys, obstacle = make_grid(ROUTING_STEP)
Xg, Yg = np.meshgrid(xs, ys)
names = list(VEHICLES.keys())


def ij_to_xy(idx):
    i, j = idx
    return float(xs[i]), float(ys[j])


init_idx = {}
for v in names:
    vx, vy = VEHICLES[v]
    init_idx[v] = (int(round((vx - xs[0]) / ROUTING_STEP)),
                   int(round((vy - ys[0]) / ROUTING_STEP)))

alert_info = {}
for a in ALERTS:
    tx, ty, tz = alert_xyz(a)
    obs = observable_mask(Xg, Yg, tx, ty, tz)
    r_h = np.hypot(tx - Xg, ty - Yg)
    az = np.abs(np.degrees(np.arctan2(ty - Yg, tx - Xg)))
    elev = np.degrees(np.arctan2(tz - CAMERA_HEIGHT, r_h))
    slew = np.maximum(az, elev) / ANG
    alert_info[a] = dict(obs=obs, slew=slew)

init_dist = {v: shortest_path_distances(VEHICLES[v], xs, ys, obstacle) for v in names}


def best_from_dist(dist, a):
    """从距离场 dist 出发, 找可观测站使 (行驶+slew) 最小。"""
    info = alert_info[a]
    ready = (dist / SPEED + info["slew"]).copy()
    ready[~info["obs"]] = np.inf
    flat = int(np.argmin(ready))
    j, i = np.unravel_index(flat, ready.shape)
    return (i, j), float(ready[j, i]), float(info["slew"][j, i])


# 固定站: 每(车,警报)从初始位置最优的站
S = {}
slew_time = {}
for v in names:
    for a in ALERTS:
        idx, _, slew = best_from_dist(init_dist[v], a)
        S[(v, a)] = idx
        slew_time[(v, a)] = slew

# 按需缓存的任意点到点绕楼行驶时间
dist_cache = {}


def travel_time(from_ij, to_ij):
    if from_ij not in dist_cache:
        dist_cache[from_ij] = shortest_path_distances(ij_to_xy(from_ij), xs, ys, obstacle)
    return dist_cache[from_ij][to_ij[1], to_ij[0]] / SPEED


# ================= 事件驱动模拟 =================
def simulate(assign, order, trace=False):
    """assign: 警报->(车1,车2); order: 车->有序警报列表。返回 (makespan, obs_start, events)."""
    veh_time = {v: 0.0 for v in names}
    veh_station = {v: None for v in names}   # None=初始位置, 否则 (i,j)
    next_task = {v: 0 for v in names}
    done = {a: False for a in ALERTS}
    obs_start = {}
    events = {v: [] for v in names}

    def ready(v, a):
        station = S[(v, a)]
        frm = veh_station[v] if veh_station[v] is not None else init_idx[v]
        tt = travel_time(frm, station)
        slew = slew_time[(v, a)]
        return veh_time[v] + tt + slew, tt, slew, station

    while not all(done.values()):
        best = None
        for a in ALERTS:
            if done[a]:
                continue
            v1, v2 = assign[a]
            if next_task[v1] >= len(order[v1]) or order[v1][next_task[v1]] != a:
                continue
            if next_task[v2] >= len(order[v2]) or order[v2][next_task[v2]] != a:
                continue
            r1, t1, s1, st1 = ready(v1, a)
            r2, t2, s2, st2 = ready(v2, a)
            start = max(r1, r2)
            if best is None or start < best[0]:
                best = (start, a, (v1, r1, t1, s1, st1), (v2, r2, t2, s2, st2))
        if best is None:
            # 死锁: 该(指派,顺序)无法构成可行时间表 -> 跳过
            return np.inf, {}, events
        start, a, i1, i2 = best
        obs_start[a] = start
        done[a] = True
        for v, r, t, s, st in [i1, i2]:
            if trace:
                dep = veh_time[v]
                events[v].append(("行驶", dep, dep + t))
                events[v].append(("转云台", dep + t, dep + t + s))
                events[v].append(("等待", r, start))
                events[v].append(("观测", start, start + OBS))
                events[v].append(("复位", start + OBS, start + OBS + s))
            veh_time[v] = start + OBS + s
            veh_station[v] = st
            next_task[v] += 1
    return max(obs_start.values()), obs_start, events


# ================= baseline: 最近指派启发式 =================
def baseline():
    veh_time = {v: 0.0 for v in names}
    veh_station = {v: None for v in names}
    assign, obs_start, events = {}, {}, {v: [] for v in names}
    for a in ALERTS:
        cand = {}
        for v in names:
            station = S[(v, a)]
            frm = veh_station[v] if veh_station[v] is not None else init_idx[v]
            tt = travel_time(frm, station)
            slew = slew_time[(v, a)]
            cand[v] = (veh_time[v] + tt + slew, tt, slew, station)
        pick = sorted(names, key=lambda v: cand[v][0])[:2]
        start = max(cand[p][0] for p in pick)
        obs_start[a] = start
        assign[a] = pick
        for p in pick:
            r, tt, slew, station = cand[p]
            dep = veh_time[p]
            events[p].append(("行驶", dep, dep + tt))
            events[p].append(("转云台", dep + tt, dep + tt + slew))
            events[p].append(("等待", r, start))
            events[p].append(("观测", start, start + OBS))
            events[p].append(("复位", start + OBS, start + OBS + slew))
            veh_time[p] = start + OBS + slew
            veh_station[p] = station
    return max(obs_start.values()), obs_start, assign, events


# ================= 优化: 枚举全部指派 × 顺序 =================
def optimize():
    pairs = list(combinations(names, 2))
    best = None   # (makespan, sum_obs_start, assign, order, obs)
    for assign_tuple in product(pairs, repeat=3):
        assign = {ALERTS[k]: assign_tuple[k] for k in range(3)}
        vtasks = {v: [] for v in names}
        for a in ALERTS:
            for v in assign[a]:
                vtasks[v].append(a)
        order_options = [list(set(permutations(vtasks[v]))) for v in names]
        for orders in product(*order_options):
            order = {v: list(orders[i]) for i, v in enumerate(names)}
            mk, obs, _ = simulate(assign, order)
            if not np.isfinite(mk):
                continue
            s = float(sum(obs.values()))
            if best is None or mk < best[0] - 1e-9 or \
               (abs(mk - best[0]) < 1e-9 and s < best[1] - 1e-9):
                best = (mk, s, assign, order, obs)
    return best


# ================= 运行 =================
b_mk, b_obs, b_assign, b_ev = baseline()
o_mk, _, o_assign, o_order, o_obs = optimize()

# ---- 输出 ----
L = []
L.append("=" * 58)
L.append("B题 问题2(2) 结果: 三警报并发车辆调度 (重写版)")
L.append("=" * 58)
L.append(f"[baseline 最近指派启发式] makespan = {b_mk:.2f} s")
for a in ALERTS:
    L.append(f"  警报 {a}: 观测开始 {b_obs[a]:.2f} s, 车辆 {b_assign[a]}")
L.append("-" * 58)
L.append(f"[优化方案] makespan = {o_mk:.2f} s  "
         f"(比 baseline 缩短 {b_mk-o_mk:.2f} s, {(b_mk-o_mk)/b_mk*100:.1f}%)")
for a in ALERTS:
    L.append(f"  警报 {a}: 观测开始 {o_obs[a]:.2f} s, 车辆 {list(o_assign[a])}")
L.append("-" * 58)
L.append("各车任务顺序:")
for v in names:
    if o_order[v]:
        L.append(f"  {v}: {' → '.join(o_order[v])}")
L.append("-" * 58)
L.append("各车观测站坐标 (绕楼最短路径):")
for v in names:
    for a in o_order[v]:
        i, j = S[(v, a)]
        L.append(f"  {v} → {a}: ({xs[i]:.1f}, {ys[j]:.1f}) m")
L.append("=" * 58)

# 最优方案时间轴
_, _, o_ev = simulate(o_assign, o_order, trace=True)
L.append("各车完整时间轴 (行驶/转云台/等待/观测/复位):")
for v in names:
    if not o_ev[v]:
        L.append(f"  {v}: 空闲")
        continue
    segs = sorted(o_ev[v], key=lambda s: s[1])
    parts = [f"{s[0]}[{s[1]:.1f}~{s[2]:.1f}]" for s in segs]
    L.append(f"  {v}: " + " → ".join(parts))
L.append("=" * 58)

print("\n".join(L))
with open(RES_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(L) + "\n")

# ================= 甘特图 =================
colors = {"行驶": "#99ccff", "转云台": "#ffcc99", "等待": "#eeeeee",
          "观测": "#ff6666", "复位": "#cccccc"}
fig, ax = plt.subplots(figsize=(12, 6))
for k, v in enumerate(names):
    for label, s, e in o_ev[v]:
        ax.broken_barh([(s, e - s)], (k - 0.35, 0.7),
                       facecolors=colors.get(label, "#ffffff"),
                       edgecolor="k", linewidth=0.4)
for a in ALERTS:
    ax.axvline(o_obs[a], color="red", ls=":", lw=1.0)
    ax.text(o_obs[a], len(names) - 0.2, a, color="red", fontsize=9, ha="center")
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names)
ax.set_xlabel("时间 / s")
ax.set_title("问题2(2): 车辆时间轴甘特图 (优化方案)")
handles = [plt.Rectangle((0, 0), 1, 1, color=colors[c]) for c in colors]
ax.legend(handles, list(colors.keys()), loc="upper right", fontsize=8)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
out_fig = os.path.join(FIG_DIR, "problem2_2_gantt.png")
plt.savefig(out_fig, dpi=150, bbox_inches="tight")
print(f"\n图已保存: {out_fig}")
print(f"结果已保存: {RES_PATH}")
plt.show()
