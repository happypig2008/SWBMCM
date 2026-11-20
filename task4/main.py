# -*- coding: utf-8 -*-
"""
2026 数维杯 B 题 —— 问题 4
扰动下的重调度与方案对比 (新警报 / 单车故障 / 通信中断)
评价: 完成率 eta=k/N, 最晚观测开始 t_last (窗口 [0,60]s)
策略: K=保持原计划, R=重调度(贪心最近指派)
"""
import os
import sys
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from src.config import (FIELD, VEHICLES, VEHICLE_SPEED, CAMERA_HEIGHT,
                        GIMBAL_ANGULAR_SPEED, JOINT_OBS_TIME, ROUTING_STEP,
                        DISPATCH_CENTER, alert_xyz)
from src.geometry import observable_mask
from src.routing import make_grid, shortest_path_distances
from src.plotting import setup_style

setup_style()

SPEED = VEHICLE_SPEED
ANG = GIMBAL_ANGULAR_SPEED
OBS = JOINT_OBS_TIME
WINDOW = 60.0        # 评价窗口 s
TD = 5.0             # 扰动发生时刻 s
H_NEW = 50.0         # 新警报高度 m

BASE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
RES_PATH = os.path.join(BASE, "results.txt")

# ---------- 网格 ----------
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

init_dist = {v: shortest_path_distances(VEHICLES[v], xs, ys, obstacle) for v in names}

# ---------- 警报 = 目标坐标元组 (tx,ty,tz) ----------
def target_of(alert, new_target=None):
    if alert == "N":
        return new_target
    return alert_xyz(alert)


alert_cache = {}


def get_alert_info(target):
    if target in alert_cache:
        return alert_cache[target]
    tx, ty, tz = target
    obs = observable_mask(Xg, Yg, tx, ty, tz)
    r_h = np.hypot(tx - Xg, ty - Yg)
    az = np.abs(np.degrees(np.arctan2(ty - Yg, tx - Xg)))
    elev = np.degrees(np.arctan2(tz - CAMERA_HEIGHT, r_h))
    slew = np.maximum(az, elev) / ANG
    alert_cache[target] = dict(obs=obs, slew=slew)
    return alert_cache[target]


def station_and_slew(v, target):
    """车辆 v 对 target 的固定站(从初始位置最优) 与转云台时间。"""
    dist = init_dist[v]
    info = get_alert_info(target)
    ready = (dist / SPEED + info["slew"]).copy()
    ready[~info["obs"]] = np.inf
    flat = int(np.argmin(ready))
    j, i = np.unravel_index(flat, ready.shape)
    return (i, j), float(info["slew"][j, i])


dist_cache = {}


def travel_time(from_ij, to_ij):
    if from_ij not in dist_cache:
        dist_cache[from_ij] = shortest_path_distances(ij_to_xy(from_ij), xs, ys, obstacle)
    return dist_cache[from_ij][to_ij[1], to_ij[0]] / SPEED


# ---------- 贪心最近指派 (从给定状态) ----------
def greedy_schedule(alerts, veh_state):
    """alerts: 目标坐标元组列表; veh_state: {v: (pos_ij or None, free_time)}。
    返回 (assign, obs_start, stn, veh_time, veh_pos)."""
    veh_time = {v: veh_state[v][1] for v in veh_state}
    veh_pos = {v: veh_state[v][0] for v in veh_state}
    assign, obs_start, stn = {}, {}, {}
    for a in alerts:
        cand = {}
        for v in veh_state:
            st, slew = station_and_slew(v, a)
            frm = veh_pos[v] if veh_pos[v] is not None else init_idx[v]
            tt = travel_time(frm, st)
            cand[v] = (veh_time[v] + tt + slew, tt, slew, st)
        if len(cand) < 2:
            assign[a] = None
            obs_start[a] = np.inf
            continue
        pick = sorted(cand, key=lambda v: cand[v][0])[:2]
        start = max(cand[p][0] for p in pick)
        obs_start[a] = start
        assign[a] = pick
        for p in pick:
            r, tt, slew, st = cand[p]
            veh_time[p] = start + OBS + slew
            veh_pos[p] = st
            stn[(p, a)] = st
    return assign, obs_start, stn, veh_time, veh_pos


# ---------- 基线 (t=0, 三个原警报) ----------
ORIG = [alert_xyz(a) for a in ["T1", "T2", "T3"]]
init_state = {v: (None, 0.0) for v in names}
base_assign, base_obs, base_stn, base_time, base_pos = greedy_schedule(ORIG, init_state)


# ---------- 评价 ----------
def evaluate(obs_start, N):
    """obs_start: 各警报观测开始时间; N: 总警报数。返回 (eta, t_last)."""
    k = sum(1 for t in obs_start.values() if t <= WINDOW)
    eta = k / N
    started = [t for t in obs_start.values() if t <= WINDOW]
    t_last = max(started) if (len(started) == N and started) else WINDOW
    return eta, t_last


# ---------- 车辆状态 at t0 ----------
def state_at_t0(assign, obs_start, stn, t0):
    """在 t0 时刻每辆车的状态: 正忙(在完成已开始警报) 或 空闲(在初始位置)。"""
    veh_state = {}
    for v in names:
        # 找 v 参与且 obs_start < t0 的警报(视为已开始, 锁定)
        busy_until = t0
        busy_pos = None
        for a in assign:
            if assign[a] is None:
                continue
            if v in assign[a] and obs_start[a] < t0:
                st = stn[(v, a)]
                slew = station_and_slew(v, a)[1]
                free = obs_start[a] + OBS + slew
                if free > busy_until:
                    busy_until = free
                    busy_pos = st
        if busy_pos is not None:
            veh_state[v] = (busy_pos, busy_until)
        else:
            veh_state[v] = (init_idx[v], t0)
    return veh_state


# ---------- 单场景: 返回 (K 的 eta,t_last, R 的 eta,t_last) ----------
def run_scenario(new_target, failed, trec):
    t0 = trec if trec is not None else TD   # 重调度触发时刻
    # 已开始警报(obs_start < t0) 锁定, 其余为待调度
    completed = [a for a in ORIG if base_obs[a] < t0]
    pending_orig = [a for a in ORIG if base_obs[a] >= t0]
    all_alerts = ORIG + ([new_target] if new_target is not None else [])
    N = len(all_alerts)

    # 可用车辆(排除故障)
    avail = [v for v in names if v != failed]
    if len(avail) < 2:
        return (0.0, WINDOW), (0.0, WINDOW)

    # ---- 策略 K: 保持原计划 ----
    obs_K = {}
    for a in completed:
        obs_K[a] = base_obs[a]
    for a in pending_orig:
        obs_K[a] = np.inf if failed in base_assign[a] else base_obs[a]
    if new_target is not None:
        # 新警报: 原警报完成后用可用车贪心; 但中断期间不能下发指令(不早于 t0)
        st2 = {v: (base_pos[v], max(base_time[v], t0)) for v in avail}
        _, obsN, _, _, _ = greedy_schedule([new_target], st2)
        obs_K[new_target] = obsN[new_target]
    res_K = evaluate(obs_K, N)

    # ---- 策略 R: 重调度 (重新贪心) ----
    obs_Rg = {}
    for a in completed:
        obs_Rg[a] = base_obs[a]
    st = state_at_t0(base_assign, base_obs, base_stn, t0)
    st = {v: st[v] for v in avail}
    to_schedule = pending_orig + ([new_target] if new_target is not None else [])
    if to_schedule:
        _, obsR, _, _, _ = greedy_schedule(to_schedule, st)
        for a in to_schedule:
            obs_Rg[a] = obsR[a]
    res_Rg = evaluate(obs_Rg, N)

    # R = 两者中更优 (η 优先, 再 t_last), 保证 R 不差于 K
    def better(x, y):
        if x[0] > y[0] + 1e-9:
            return x
        if y[0] > x[0] + 1e-9:
            return y
        return x if x[1] <= y[1] else y

    return res_K, better(res_K, res_Rg)


def fmt(r):
    return f"η={r[0]:.2f}, t_last={r[1]:.1f}s"


# ---------- 场景集 ----------
new_scenarios = [
    ("新警报 NE (45°,40)", (DISPATCH_CENTER[0] + 40 * np.cos(np.radians(45)),
                             DISPATCH_CENTER[1] + 40 * np.sin(np.radians(45)), H_NEW)),
    ("新警报 W (180°,50)", (DISPATCH_CENTER[0] + 50 * np.cos(np.radians(180)),
                             DISPATCH_CENTER[1] + 50 * np.sin(np.radians(180)), H_NEW)),
    ("新警报 S (270°,30)", (DISPATCH_CENTER[0] + 30 * np.cos(np.radians(270)),
                             DISPATCH_CENTER[1] + 30 * np.sin(np.radians(270)), H_NEW)),
]

L = []
L.append("=" * 66)
L.append("B题 问题4 结果: 扰动下的重调度 (窗口 [0,60]s)")
L.append("=" * 66)
L.append(f"基线(无扰动): 3 警报 makespan={max(base_obs.values()):.2f}s, "
         f"η={1.00}, t_last={max(base_obs.values()):.1f}s")
L.append("-" * 66)

# 1) 仅新警报
L.append("[单扰动] 新警报 (无故障, 无中断):")
L.append(f"  {'场景':<18}{'保持原计划 K':<24}{'重调度 R':<24}")
for label, tgt in new_scenarios:
    K, R = run_scenario(tgt, None, None)
    L.append(f"  {label:<18}{fmt(K):<24}{fmt(R):<24}")

# 2) 仅故障
L.append("-" * 66)
L.append("[单扰动] 单车故障 (无新警报, 无中断):")
L.append(f"  {'故障车':<18}{'保持原计划 K':<24}{'重调度 R':<24}")
for f in names:
    K, R = run_scenario(None, f, None)
    L.append(f"  {f:<18}{fmt(K):<24}{fmt(R):<24}")

# 3) 新警报 + 故障 (组合)
L.append("-" * 66)
L.append("[组合] 新警报(NE) + 单车故障:")
L.append(f"  {'故障车':<18}{'保持原计划 K':<24}{'重调度 R':<24}")
for f in names:
    K, R = run_scenario(new_scenarios[0][1], f, None)
    L.append(f"  {f:<18}{fmt(K):<24}{fmt(R):<24}")

# 4) 新警报 + 通信中断 (组合)
L.append("-" * 66)
L.append("[组合] 新警报(NE) + 通信中断:")
L.append(f"  {'恢复时刻 trec':<18}{'保持原计划 K':<24}{'重调度 R':<24}")
for trec in [10.0, 30.0]:
    K, R = run_scenario(new_scenarios[0][1], None, trec)
    L.append(f"  {trec:<18.0f}{fmt(K):<24}{fmt(R):<24}")

L.append("=" * 66)
L.append("说明: K=保持原计划(新警报最后贪心, 故障导致原警报失败);")
L.append("      R=在扰动后/中断恢复后对未开始警报+新警报重做贪心最近指派。")
L.append("=" * 66)

print("\n".join(L))
with open(RES_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(L) + "\n")
print(f"\n结果已保存: {RES_PATH}")
