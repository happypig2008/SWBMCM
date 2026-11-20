# -*- coding: utf-8 -*-
"""
2026 数维杯 B 题 —— 问题 1 (最终版)
可观测区域 + 单目标最优观测站 (T1)
方法: 解析下界 r_min + 半径 r_min 圆弧上的 1D 搜索 (破平选代表点)
输出: 结果文本 + 4 宫格图 (区域图 / 局部解释图 / 3D 视图 / 仰角-距离曲线)
"""
import os
import sys

# 让运行脚本时能 import src 包 (把项目根目录加入搜索路径)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from src.config import (FIELD, DISPATCH_CENTER, BUILDINGS, CAMERA_HEIGHT,
                        GIMBAL_ELEV, VEHICLE_RADIUS, alert_xyz)
from src.geometry import observable_mask, dist_to_rect, elevation_deg

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "SimSun"]
matplotlib.rcParams["axes.unicode_minus"] = False

# ---------- 目标与输出目录 ----------
TARGET = "T1"
Tx, Ty, Tz = alert_xyz(TARGET)
cx, cy = DISPATCH_CENTER
elev_min, elev_max = GIMBAL_ELEV

BASE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
RES_PATH = os.path.join(BASE, f"results_{TARGET}.txt")

# ---------- 1) 解析下界 r_min ----------
# 仰角 = arctan((Tz-CAMERA_HEIGHT)/r) <= elev_max  =>  r >= (Tz-CAMERA_HEIGHT)/tan(elev_max)
r_min = (Tz - CAMERA_HEIGHT) / np.tan(np.radians(elev_max))

# ---------- 2) 全场地可观测区域 (网格, 供绘图) ----------
step = 0.5
xs = np.arange(FIELD[0], FIELD[1] + step, step)
ys = np.arange(FIELD[2], FIELD[3] + step, step)
Xg, Yg = np.meshgrid(xs, ys)
region = observable_mask(Xg, Yg, Tx, Ty, Tz)

# ---------- 3) 最优观测站: 半径 r_min 圆上的 1D 搜索 ----------
th = np.linspace(0.0, 2.0 * np.pi, 360000, endpoint=False)
Xc = Tx + r_min * np.cos(th)
Yc = Ty + r_min * np.sin(th)
feas = observable_mask(Xc, Yc, Tx, Ty, Tz, elev_tol=0.01)  # 容差吸收浮点误差


def pick_rep(key, want_max=False):
    """在可行点中按 key 取最优 (最小或最大), 返回 (x, y)。"""
    v = key.astype(float).copy()
    if want_max:
        v[~feas] = -np.inf
        i = int(np.argmax(v))
    else:
        v[~feas] = np.inf
        i = int(np.argmin(v))
    return float(Xc[i]), float(Yc[i])


# 破平规则1(选定): 离调度中心最近 -> 行驶/部署成本最低
dcenter = np.hypot(Xc - cx, Yc - cy)
bx, by = pick_rep(dcenter)

# 破平规则2(备选): 离建筑最远 -> 安全余量最大
clear = np.full_like(Xc, np.inf)
for _, xmin, xmax, ymin, ymax, _ in BUILDINGS:
    clear = np.minimum(clear, dist_to_rect(Xc, Yc, xmin, xmax, ymin, ymax))
ax2, ay2 = pick_rep(clear, want_max=True)

r_best = float(np.hypot(Tx - bx, Ty - by))
elev_best = float(elevation_deg(r_best, Tz))
d_best = float(np.hypot(bx - cx, by - cy))

# 可行圆弧的角范围 (度)
def contiguous_arcs(feas):
    idx = np.where(feas)[0]
    if len(idx) == 0:
        return []
    arcs, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i == prev + 1:
            prev = i
        else:
            arcs.append((start, prev)); start = prev = i
    arcs.append((start, prev))
    if len(arcs) > 1 and arcs[0][0] == 0 and arcs[-1][1] == len(feas) - 1:
        arcs[0] = (arcs[-1][0], arcs[0][1]); arcs = arcs[:-1]
    return [(a / len(feas) * 360.0, b / len(feas) * 360.0) for a, b in arcs]

arcs = contiguous_arcs(feas)
arc_desc = "; ".join(f"[{a:.1f}°, {b:.1f}°]" for a, b in arcs) if arcs else "无"

# ---------- 输出 ----------
lines = [
    "=" * 48,
    f"B题 问题1 结果 (目标 {TARGET})",
    "=" * 48,
    f"目标 {TARGET} 三维坐标: ({Tx:.3f}, {Ty:.3f}, {Tz}) m",
    f"解析下界 r_min = (Tz-{CAMERA_HEIGHT})/tan({elev_max}°) = {r_min:.4f} m",
    f"全场地可观测格点数: {region.sum()} / {region.size} (步长 {step} m)",
    f"最优圆弧上可行角点占比: {feas.mean()*100:.2f}%",
    f"可行圆弧(仰角=80°处)角度区间: {arc_desc}",
    "-" * 48,
    f"[选定破平规则] 离调度中心最近:",
    f"  最优观测站 = ({bx:.4f}, {by:.4f}) m",
    f"  水平距离 r = {r_best:.4f} m   (观测质量 Q=1/r = {1/r_best:.4f})",
    f"  视线仰角   = {elev_best:.3f}°  (云台范围 [{elev_min}, {elev_max}]°)",
    f"  距调度中心 = {d_best:.4f} m",
    f"[备选破平规则] 离建筑最远(最大安全余量): ({ax2:.4f}, {ay2:.4f}) m",
    "-" * 48,
    "说明: 仰角约束是唯一紧约束, 使 r 恒等于 r_min;",
    "因此在半径 r_min 圆上存在一整段『等价最优』圆弧,",
    "破平规则仅用于从中选出唯一的代表点。",
    "=" * 48,
]
print("\n".join(lines))
with open(RES_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

# ===================== 绘图 =====================
fig = plt.figure(figsize=(14, 12))

# 自动缩放窗口 (对 T1/T2/T3 通用)
zoom_half = max(3.0 * r_min, 15.0)
zx_min = max(FIELD[0], Tx - zoom_half)
zx_max = min(FIELD[1], Tx + zoom_half)
zy_min = max(FIELD[2], Ty - zoom_half)
zy_max = min(FIELD[3], Ty + zoom_half)


def in_zoom(xmin, xmax, ymin, ymax):
    return not (xmax < zx_min or xmin > zx_max or ymax < zy_min or ymin > zy_max)


# ---- 面板1: 全场地可观测区域 ----
ax1 = fig.add_subplot(2, 2, 1)
ax1.pcolormesh(xs, ys, region.T, cmap="Greens", alpha=0.35, shading="auto")
for name, xmin, xmax, ymin, ymax, h in BUILDINGS:
    ax1.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                            facecolor="gray", edgecolor="k", lw=1.2))
    ax1.text((xmin + xmax) / 2, (ymin + ymax) / 2, f"{name}\n{h}m",
             ha="center", va="center", fontsize=8, color="white")
ax1.plot(cx, cy, "ko", ms=6); ax1.text(cx, cy - 2.5, "O(50,50)", ha="center", fontsize=9)
ax1.plot(Tx, Ty, "r^", ms=11)
ax1.text(Tx, Ty + 2.5, f"{TARGET}\n({Tx:.1f},{Ty:.1f},{Tz})", ha="center", fontsize=9, color="red")
ax1.plot(bx, by, "b*", ms=16, label="代表点(距中心最近)")
ax1.plot(ax2, ay2, "ms", ms=10, label="备选点(距建筑最远)")
ax1.plot([bx, Tx], [by, Ty], "b--", lw=1.2, label="视线(水平投影)")
ax1.set_xlim(FIELD[0], FIELD[1]); ax1.set_ylim(FIELD[2], FIELD[3])
ax1.set_xlabel("x / m (东)"); ax1.set_ylabel("y / m (北)")
ax1.set_title("(a) 全场地可观测区域")
ax1.legend(loc="lower left", fontsize=8); ax1.grid(alpha=0.3); ax1.set_aspect("equal")

# ---- 面板2: 局部解释图 (仰角极限圆 + 建筑缓冲 + 可行弧) ----
ax2_ = fig.add_subplot(2, 2, 2)
first_buf = True
for name, xmin, xmax, ymin, ymax, h in BUILDINGS:
    if not in_zoom(xmin, xmax, ymin, ymax):
        continue
    ax2_.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                             facecolor="gray", edgecolor="k", lw=1.2, zorder=2))
    ax2_.text((xmin + xmax) / 2, (ymin + ymax) / 2, f"{name}({h}m)",
              ha="center", va="center", fontsize=8, color="white", zorder=3)
    buf = FancyBboxPatch((xmin - VEHICLE_RADIUS, ymin - VEHICLE_RADIUS),
                         xmax - xmin + 2 * VEHICLE_RADIUS,
                         ymax - ymin + 2 * VEHICLE_RADIUS,
                         boxstyle="round,pad=0,rounding_size=%.2f" % VEHICLE_RADIUS,
                         fc="none", ec="orange", ls="--", lw=1.2, zorder=2,
                         label="建筑 1.5m 缓冲" if first_buf else None)
    first_buf = False
    ax2_.add_patch(buf)
ax2_.add_patch(Circle((Tx, Ty), r_min, fc="none", ec="green", lw=1.6, zorder=4,
                      label="仰角极限圆 r=r_min"))
# 可行弧 (加粗)
ax2_.plot(Xc[feas], Yc[feas], "g.", ms=0.6, zorder=4)
ax2_.plot(Tx, Ty, "r^", ms=11, zorder=5)
ax2_.text(Tx + 0.8, Ty + 0.8, TARGET, fontsize=9, color="red", zorder=5)
ax2_.plot(bx, by, "b*", ms=16, zorder=6)
ax2_.text(bx + 0.6, by + 0.6, "代表点", fontsize=8, color="blue", zorder=6)
ax2_.plot(ax2, ay2, "ms", ms=10, zorder=6)
ax2_.plot([bx, Tx], [by, Ty], "b--", lw=1.2, zorder=5)
ax2_.set_xlim(zx_min, zx_max); ax2_.set_ylim(zy_min, zy_max)
ax2_.set_xlabel("x / m"); ax2_.set_ylabel("y / m")
ax2_.set_title("(b) 局部解释: 仰角极限圆 + 建筑缓冲 + 最优弧")
ax2_.legend(loc="best", fontsize=8)
ax2_.grid(alpha=0.3); ax2_.set_aspect("equal")

# ---- 面板3: 3D 视图 ----
ax3 = fig.add_subplot(2, 2, 3, projection="3d")


def draw_box_3d(ax, xmin, xmax, ymin, ymax, zmin, zmax, **kw):
    v = [(x, y, z) for x in (xmin, xmax) for y in (ymin, ymax) for z in (zmin, zmax)]
    faces = [[v[0], v[2], v[6], v[4]], [v[1], v[3], v[7], v[5]],
             [v[0], v[1], v[5], v[4]], [v[2], v[3], v[7], v[6]],
             [v[0], v[1], v[3], v[2]], [v[4], v[5], v[7], v[6]]]
    ax.add_collection3d(Poly3DCollection(faces, **kw))


for name, xmin, xmax, ymin, ymax, h in BUILDINGS:
    draw_box_3d(ax3, xmin, xmax, ymin, ymax, 0, h,
                facecolor="gray" if name != "B2" else "darkgray",
                edgecolor="k", alpha=0.55 if name != "B2" else 0.95)
ax3.plot([bx, Tx], [by, Ty], [CAMERA_HEIGHT, Tz], "r-", lw=2.5, label="视线")
ax3.plot([bx], [by], [CAMERA_HEIGHT], "bo", ms=7, label="相机")
ax3.plot([Tx], [Ty], [Tz], "r^", ms=8, label="目标 T1")
ax3.set_xlabel("x / m"); ax3.set_ylabel("y / m"); ax3.set_zlabel("z / m")
ax3.set_title("(c) 三维视线 (仰角≈80°)")
ax3.set_xlim(zx_min, zx_max); ax3.set_ylim(zy_min, zy_max); ax3.set_zlim(0, Tz + 5)
ax3.view_init(elev=18, azim=-60)
ax3.legend(fontsize=8)

# ---- 面板4: 仰角-距离关系 ----
ax4 = fig.add_subplot(2, 2, 4)
r_vals = np.linspace(2.0, 60.0, 500)
elev_vals = elevation_deg(r_vals, Tz)
ax4.plot(r_vals, elev_vals, "b-", lw=2)
ax4.axhline(elev_max, color="r", ls="--", lw=1.5, label=f"云台上界 {elev_max}°")
ax4.axhline(elev_min, color="orange", ls="--", lw=1.5, label=f"云台下界 {elev_min}°")
ax4.axvline(r_min, color="g", ls="--", lw=1.5)
ax4.plot([r_min], [elev_max], "ko", ms=9)
ax4.annotate(f"最优 (r={r_min:.2f}, {elev_max}°)",
             xy=(r_min, elev_max), xytext=(r_min + 8, elev_max - 12),
             arrowprops=dict(arrowstyle="->", color="k"))
ax4.set_xlabel("水平距离 r / m"); ax4.set_ylabel("视线仰角 / °")
ax4.set_title("(d) 仰角—距离关系 (决定 r_min)")
ax4.grid(alpha=0.3); ax4.legend(fontsize=8)

plt.tight_layout()
out_fig = os.path.join(FIG_DIR, f"problem1_{TARGET}.png")
plt.savefig(out_fig, dpi=150)
print(f"\n图已保存: {out_fig}")
print(f"结果已保存: {RES_PATH}")
plt.show()
