# -*- coding: utf-8 -*-
"""
2026 数维杯 B 题 —— 问题 2(1) 优化版
- 绕楼最短路径 (网格 Dijkstra) 替代欧氏距离
- 粗网格枚举 (1m) + 局部精化 (0.1m 坐标下降)
- 复用 src/routing.py + src/plotting.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from src.config import (FIELD, BUILDINGS, VEHICLES, VEHICLE_SPEED, CAMERA_HEIGHT,
                        ROUTING_STEP, REFINE_STEP, R_CUT, T_MAX, SIGMA_AZ_DEG,
                        alert_xyz)
from src.geometry import observable_mask
from src.routing import make_grid, shortest_path_distances
from src.plotting import setup_style, draw_buildings
from matplotlib.patches import Ellipse

setup_style()

TARGET = "T2"
Tx, Ty, Tz = alert_xyz(TARGET)
SIGMA_AZ = np.radians(SIGMA_AZ_DEG)
D_MAX = VEHICLE_SPEED * T_MAX

BASE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
RES_PATH = os.path.join(BASE, f"results_{TARGET}.txt")

# ---- 网格 / 可观测 / 绕楼距离场 ----
xs, ys, obstacle = make_grid(ROUTING_STEP)
Xg, Yg = np.meshgrid(xs, ys)
obs = observable_mask(Xg, Yg, Tx, Ty, Tz)
near = (np.abs(Xg - Tx) <= R_CUT) & (np.abs(Yg - Ty) <= R_CUT)

names = list(VEHICLES.keys())
dist_fields = {n: shortest_path_distances(VEHICLES[n], xs, ys, obstacle) for n in names}
feas = {n: obs & near & (dist_fields[n] <= D_MAX) for n in names}


def gdop(x1, y1, x2, y2):
    """GDOP = sqrt(r1^2+r2^2) / |sin(gamma)|, 支持广播。"""
    r1 = np.hypot(Tx - x1, Ty - y1)
    r2 = np.hypot(Tx - x2, Ty - y2)
    b1 = np.arctan2(Ty - y1, Tx - x1)
    b2 = np.arctan2(Ty - y2, Tx - x2)
    sin_g = np.clip(np.abs(np.sin(b2 - b1)), 1e-12, None)
    return np.sqrt(r1 ** 2 + r2 ** 2) / sin_g


def feasible_xy(name):
    m = feas[name]
    return Xg[m], Yg[m], dist_fields[name][m] / VEHICLE_SPEED


fxy = {n: feasible_xy(n) for n in names}

# ---- 粗枚举 (1m 网格) ----
best = None
for a in range(len(names)):
    for b in range(a + 1, len(names)):
        na, nb = names[a], names[b]
        xa, ya, ta = fxy[na]
        xb, yb, tb = fxy[nb]
        if xa.size == 0 or xb.size == 0:
            continue
        G = gdop(xa[:, None], ya[:, None], xb[None, :], yb[None, :])
        gmin = float(G.min())
        sel = np.isclose(G, gmin, atol=1e-6)
        T = ta[:, None] + tb[None, :]
        T[~sel] = np.inf
        idx = np.unravel_index(int(np.argmin(T)), T.shape)
        cand = (gmin, float(T[idx]), na, nb,
                float(xa[idx[0]]), float(ya[idx[0]]),
                float(xb[idx[1]]), float(yb[idx[1]]))
        if best is None or cand[0] < best[0] - 1e-9 or \
           (abs(cand[0] - best[0]) < 1e-9 and cand[1] < best[1]):
            best = cand

gval, ttot, na, nb, x1, y1, x2, y2 = best

# ---- 局部精化 (坐标下降, 0.1m) ----
def refine_station(x_other, y_other, xc, yc, half=1.0):
    step = REFINE_STEP
    xx = np.arange(xc - half, xc + half + step * 0.5, step)
    yy = np.arange(yc - half, yc + half + step * 0.5, step)
    XX, YY = np.meshgrid(xx, yy)
    ok = observable_mask(XX, YY, Tx, Ty, Tz)
    G = gdop(XX, YY, x_other, y_other)
    G[~ok] = np.inf
    j, i = np.unravel_index(int(np.argmin(G)), G.shape)
    return float(xx[i]), float(yy[j])


for _ in range(4):
    x1, y1 = refine_station(x2, y2, x1, y1)
    x2, y2 = refine_station(x1, y1, x2, y2)

gval = float(gdop(np.array([x1]), np.array([y1]),
                  np.array([x2]), np.array([y2]))[0])


def travel_time(name, x, y):
    xi = int(np.clip(round((x - xs[0]) / ROUTING_STEP), 0, len(xs) - 1))
    yi = int(np.clip(round((y - ys[0]) / ROUTING_STEP), 0, len(ys) - 1))
    return float(dist_fields[name][yi, xi] / VEHICLE_SPEED)


t1 = travel_time(na, x1, y1)
t2 = travel_time(nb, x2, y2)

# ---- 结果量 ----
r1 = float(np.hypot(Tx - x1, Ty - y1))
r2 = float(np.hypot(Tx - x2, Ty - y2))
b1 = np.arctan2(Ty - y1, Tx - x1)
b2 = np.arctan2(Ty - y2, Tx - x2)
gamma = float(np.degrees(np.arccos(np.clip(np.cos(b1) * np.cos(b2) +
                                           np.sin(b1) * np.sin(b2), -1, 1))))
baseline = float(np.hypot(x1 - x2, y1 - y2))
pos_err = SIGMA_AZ * gval
pos_err95 = 1.96 * pos_err

# ---- 误差椭圆 (2x2 协方差矩阵的特征值与方向) ----
H = np.array([[-np.sin(b1) / r1, np.cos(b1) / r1],
              [-np.sin(b2) / r2, np.cos(b2) / r2]])
C = SIGMA_AZ ** 2 * np.linalg.inv(H.T @ H)
lam, V = np.linalg.eigh(C)                       # 升序特征值, V 列=特征向量
semi = np.sqrt(lam)                              # 1σ 半轴 (m)
semi_major, semi_minor = semi[1], semi[0]
phi = float(np.degrees(np.arctan2(V[1, 1], V[0, 1])))   # 长轴方向角
aniso = semi_major / semi_minor                  # 各向异性比
semi95 = semi * 2.448                            # 95% 置信椭圆半轴 (χ²_2(0.95)=5.991)

r_min_theory = (Tz - CAMERA_HEIGHT) / np.tan(np.radians(80.0))
gdop_theory = np.sqrt(2.0) * r_min_theory

lines = [
    "=" * 52,
    f"B题 问题2(1) 优化版 结果 (目标 {TARGET})",
    "=" * 52,
    f"目标 {TARGET} 三维坐标: ({Tx:.3f}, {Ty:.3f}, {Tz}) m",
    f"方位角误差 σ = {SIGMA_AZ_DEG}° (独立同分布)",
    "-" * 52,
    f"最优车辆对: {na} + {nb}",
    f"  观测站 {na}: ({x1:.3f}, {y1:.3f}) m   行驶时间 {t1:.2f} s",
    f"  观测站 {nb}: ({x2:.3f}, {y2:.3f}) m   行驶时间 {t2:.2f} s",
    f"  站间基线长 B = {baseline:.3f} m",
    f"  视线距离 r1 = {r1:.3f} m, r2 = {r2:.3f} m",
    f"  交汇角 γ = {gamma:.2f}°",
    "-" * 52,
    f"精度指标 GDOP = {gval:.4f} m/rad  (理论下界 √2·r_min = {gdop_theory:.4f})",
    f"定位误差 1σ = {pos_err*100:.2f} cm,  95% ≈ {pos_err95*100:.2f} cm",
    "-" * 52,
    f"误差椭圆(1σ): 长半轴 {semi_major*100:.2f} cm, 短半轴 {semi_minor*100:.2f} cm",
    f"  长轴方向 {phi:.1f}°, 各向异性比(长/短) = {aniso:.2f}",
    f"误差椭圆(95%): 长半轴 {semi95[1]*100:.2f} cm, 短半轴 {semi95[0]*100:.2f} cm",
    "=" * 52,
    "行驶距离: 绕楼最短路径 (网格 Dijkstra, 8 邻接)",
    "搜索: 1m 粗枚举 + 0.1m 坐标下降精化",
    "=" * 52,
]
print("\n".join(lines))
with open(RES_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

# ===================== 绘图 =====================
fig = plt.figure(figsize=(13, 12))
gs = fig.add_gridspec(2, 2)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])
ax4.axis("off")

# ---- 面板1: 全场地 ----
draw_buildings(ax1)
for n, (vx, vy) in VEHICLES.items():
    ax1.plot(vx, vy, "s", ms=8, mfc="white", mec="k")
    ax1.text(vx, vy + 2.0, n, ha="center", fontsize=8)
ax1.plot(Tx, Ty, "r^", ms=11)
ax1.text(Tx, Ty + 2.0, TARGET, ha="center", fontsize=9, color="red")
ax1.plot([x1, x2], [y1, y2], "b*", ms=13)
ax1.plot([x1, Tx], [y1, Ty], "b--", lw=1.2)
ax1.plot([x2, Tx], [y2, Ty], "g--", lw=1.2)
ax1.plot([x1, x2], [y1, y2], "k-.", lw=1.0)
ax1.text((x1 + x2) / 2, (y1 + y2) / 2 - 2.5, f"基线 B={baseline:.1f}m",
         ha="center", fontsize=8)
ax1.set_xlim(FIELD[0], FIELD[1]); ax1.set_ylim(FIELD[2], FIELD[3])
ax1.set_xlabel("x / m"); ax1.set_ylabel("y / m")
ax1.set_title(f"(a) 全场地: 最优站对 ({na},{nb})")
ax1.grid(alpha=0.3); ax1.set_aspect("equal")

# ---- 面板2: 局部放大 ----
draw_buildings(ax2)
ax2.plot(Tx, Ty, "r^", ms=11)
ax2.text(Tx + 0.3, Ty + 0.3, TARGET, fontsize=9, color="red")
ax2.plot([x1, x2], [y1, y2], "b*", ms=13)
ax2.text(x1 - 1.5, y1 - 1.5, na, fontsize=8, color="blue")
ax2.text(x2 + 1.0, y2 + 0.5, nb, fontsize=8, color="blue")
ax2.plot([x1, Tx], [y1, Ty], "b--", lw=1.5)
ax2.plot([x2, Tx], [y2, Ty], "g--", lw=1.5)
ax2.annotate(f"γ={gamma:.1f}°", xy=(Tx, Ty), xytext=(Tx - 6, Ty - 4),
             arrowprops=dict(arrowstyle="->", color="k"), fontsize=9)
ax2.set_xlim(Tx - R_CUT, Tx + R_CUT); ax2.set_ylim(Ty - R_CUT, Ty + R_CUT)
ax2.set_xlabel("x / m"); ax2.set_ylabel("y / m")
ax2.set_title(f"(b) 局部: 交汇角 γ={gamma:.1f}°")
ax2.grid(alpha=0.3); ax2.set_aspect("equal")

# ---- 面板3: 误差椭圆 (放大到目标附近) ----
zoom = 1.8 * semi95[1]
ax3.add_patch(Ellipse((Tx, Ty), 2 * semi95[1], 2 * semi95[0], angle=phi,
                      facecolor="purple", alpha=0.28, edgecolor="purple", lw=2,
                      label="95% 误差椭圆"))
ax3.plot(Tx, Ty, "r^", ms=11, zorder=5)
ax3.text(Tx, Ty + zoom * 0.16, TARGET, fontsize=10, color="red",
         ha="center", zorder=5)
# 比例尺 (10 cm)
ax3.plot([Tx + 0.55 * zoom, Tx + 0.55 * zoom + 0.1],
         [Ty - 0.82 * zoom, Ty - 0.82 * zoom], "k-", lw=2.5)
ax3.text(Tx + 0.55 * zoom + 0.05, Ty - 0.70 * zoom, "10 cm",
         fontsize=8, ha="center")
ax3.set_xlim(Tx - zoom, Tx + zoom); ax3.set_ylim(Ty - zoom, Ty + zoom)
ax3.set_xlabel("x / m"); ax3.set_ylabel("y / m")
ax3.set_title(f"(c) 95% 误差椭圆\n长半轴 {semi95[1]*100:.1f} cm, "
              f"短半轴 {semi95[0]*100:.1f} cm (近似圆形)")
ax3.legend(fontsize=8, loc="upper right")
ax3.grid(alpha=0.35, ls=":")
ax3.set_aspect("equal")

# ---- 面板4: 结果摘要 ----
ax4.text(0.05, 0.97, "问题2(1) 最优解摘要", transform=ax4.transAxes,
         fontsize=14, va="top", ha="left")
body = (
    f"最优车辆对：{na} + {nb}\n"
    f"观测站 {na}：({x1:.2f}, {y1:.2f}) m\n"
    f"观测站 {nb}：({x2:.2f}, {y2:.2f}) m\n"
    f"基线长 B = {baseline:.2f} m，交汇角 γ = {gamma:.2f}°\n"
    f"视线距离 r1 = {r1:.2f} m，r2 = {r2:.2f} m\n\n"
    f"GDOP = {gval:.4f} m/rad\n"
    f"定位误差 1σ = {pos_err*100:.1f} cm，95% = {pos_err95*100:.1f} cm\n\n"
    f"误差椭圆(95%)：\n"
    f"  长半轴 {semi95[1]*100:.1f} cm，短半轴 {semi95[0]*100:.1f} cm\n"
    f"  各向异性比 {aniso:.3f}（近似圆形）"
)
ax4.text(0.05, 0.90, body, transform=ax4.transAxes,
         fontsize=11, va="top", ha="left", linespacing=1.8)

plt.tight_layout()
out_fig = os.path.join(FIG_DIR, f"problem2_1_{TARGET}.png")
plt.savefig(out_fig, dpi=150, bbox_inches="tight")
print(f"\n图已保存: {out_fig}")
print(f"结果已保存: {RES_PATH}")
plt.show()
