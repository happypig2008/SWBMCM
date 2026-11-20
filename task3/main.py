# -*- coding: utf-8 -*-
"""
2026 数维杯 B 题 —— 问题 3
定位误差 + 云台指向误差 对水平方位精度的影响
- 解析: 一阶误差传播 (σ_Δψ^2 = σ_pos^2/r^2 + σ0^2 + σc^2)
- 验证: 蒙特卡洛 (精确 atan2, 非近似)
- 求解: s_max (使 P(|Δψ|<=5°)>=0.95) 与 a+80b 约束
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from src.plotting import setup_style

setup_style()

BASE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
RES_PATH = os.path.join(BASE, "results.txt")

# ---------- 参数 ----------
R = 40.0            # 车到目标水平距离 m
SIGMA0 = 0.8        # 云台零偏 (度)
SIGMAC = 0.6        # 云台控制误差 (度)
THRESH = 5.0        # 允许的方位偏差 (度)
PROB = 0.95         # 置信水平
Z = 1.959964        # 标准正态 97.5% 分位点 (对应 95% 双侧)

SIGMA0_R = np.radians(SIGMA0)
SIGMAC_R = np.radians(SIGMAC)
THRESH_R = np.radians(THRESH)

# ---------- 解析求解 ----------
sigma_max = THRESH_R / Z                      # 允许的最大 σ_Δψ (rad)
var_gimbal = SIGMA0_R ** 2 + SIGMAC_R ** 2    # 云台两项方差和 (rad^2)
# (σ_pos/R)^2 + var_gimbal <= sigma_max^2  =>  σ_pos <= R*sqrt(sigma_max^2 - var_gimbal)
sigma_pos_max = R * np.sqrt(sigma_max ** 2 - var_gimbal)

s_max = (sigma_pos_max - 0.3) / 0.02          # 由 σ_pos = 0.3 + 0.02s 反解
rhs_80 = sigma_pos_max                        # 约束: a + 80*b <= rhs_80

L = []
L.append("=" * 52)
L.append("B题 问题3 结果: 方位偏差误差传播")
L.append("=" * 52)
L.append(f"车到目标水平距离 r = {R} m (固定)")
L.append(f"云台零偏 σ0 = {SIGMA0}°, 控制误差 σc = {SIGMAC}°")
L.append(f"要求: P(|Δψ| <= {THRESH}°) >= {PROB}  (Z = {Z})")
L.append("-" * 52)
L.append(f"允许的最大 σ_Δψ = {THRESH}/{Z} = {np.degrees(sigma_max):.3f}°")
L.append(f"云台两项方差和 = {SIGMA0**2 + SIGMAC**2:.2f} deg^2 (即 1.0°)")
L.append(f"最大允许定位标准差 σ_pos = {sigma_pos_max:.4f} m")
L.append("-" * 52)
L.append(f"[1] 最大允许行驶距离 s_max = {s_max:.2f} m")
L.append(f"[2] 行驶 80m 时约束: a + 80*b <= {rhs_80:.4f} m (a>=0, b>=0)")
L.append("=" * 52)

# ---------- 蒙特卡洛验证 ----------
def mc_prob(s, a=0.3, b=0.02, N=300000, seed=0):
    rng = np.random.default_rng(seed)
    sigma_pos = a + b * s
    dx = rng.normal(0.0, sigma_pos, N)
    dy = rng.normal(0.0, sigma_pos, N)
    # 目标固定于 (R, 0), 车真实位置在原点 -> 真实方位角 0
    psi_c = np.arctan2(0.0 - dy, R - dx)          # 用估计位置算的指令方位角
    eps0 = rng.normal(0.0, SIGMA0_R, N)
    epsc = rng.normal(0.0, SIGMAC_R, N)
    dpsi = psi_c + eps0 + epsc                    # psi_t = 0
    dpsi = np.arctan2(np.sin(dpsi), np.cos(dpsi)) # wrap 到 [-π, π]
    return float(np.mean(np.abs(dpsi) <= THRESH_R))

L.append("蒙特卡洛验证 (精确 atan2, N=300000):")
for s in [0.0, 30.0, s_max, 80.0]:
    p = mc_prob(s)
    L.append(f"  s = {s:6.2f} m : P(|Δψ|<=5°) ≈ {p:.4f}")
L.append("=" * 52)
L.append("误差源相对贡献 (方差占比):")
for label, var_deg2 in [("定位误差", (np.degrees(sigma_pos_max / R) ** 2)),
                         ("云台零偏", SIGMA0 ** 2), ("云台控制", SIGMAC ** 2)]:
    total = (np.degrees(sigma_pos_max / R) ** 2) + SIGMA0 ** 2 + SIGMAC ** 2
    L.append(f"  {label}: {var_deg2 / total * 100:.1f}%  (在 s_max 处)")
L.append("=" * 52)
L.append("部署建议:")
L.append("  1) 小 s 时云台零偏主导, 大 s 时定位误差主导 (随 s 二次增长);")
L.append(f"  2) 为满足 95% 要求, 行驶距离不宜超过 {s_max:.1f} m;")
L.append(f"  3) 若必须行驶 80 m, 需收紧定位误差模型至 a+80b <= {rhs_80:.3f} m。")
L.append("=" * 52)

print("\n".join(L))
with open(RES_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(L) + "\n")

# ================= 绘图 =================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# 面板1: σ_Δψ 随 s 变化
ss = np.linspace(0, 100, 400)
sigma_pos_s = 0.3 + 0.02 * ss
sigma_psi_pos_deg = np.degrees(sigma_pos_s / R)
sigma_dpsi_deg = np.sqrt(sigma_psi_pos_deg ** 2 + SIGMA0 ** 2 + SIGMAC ** 2)
ax1.plot(ss, sigma_dpsi_deg, "b-", lw=2, label="σ_Δψ")
ax1.axhline(np.degrees(sigma_max), color="r", ls="--", lw=1.5,
            label=f"阈值 {THRESH}°/Z = {np.degrees(sigma_max):.2f}°")
ax1.axvline(s_max, color="g", ls="--", lw=1.5)
ax1.plot([s_max], [np.degrees(sigma_max)], "ko", ms=8)
ax1.annotate(f"s_max={s_max:.1f}m", xy=(s_max, np.degrees(sigma_max)),
             xytext=(s_max + 5, np.degrees(sigma_max) - 0.5),
             arrowprops=dict(arrowstyle="->", color="k"))
ax1.set_xlabel("累计行驶距离 s / m")
ax1.set_ylabel("方位偏差标准差 σ_Δψ / °")
ax1.set_title("(a) σ_Δψ 随 s 变化")
ax1.grid(alpha=0.3); ax1.legend(fontsize=8)

# 面板2: 三项误差源方差占比随 s 变化
var_pos = sigma_psi_pos_deg ** 2
var0 = SIGMA0 ** 2 * np.ones_like(ss)
varc = SIGMAC ** 2 * np.ones_like(ss)
tot = var_pos + var0 + varc
ax2.stackplot(ss, var_pos / tot * 100, var0 / tot * 100, varc / tot * 100,
              labels=["定位误差", "云台零偏", "云台控制"],
              colors=["#4c72b0", "#dd8452", "#55a868"], alpha=0.8)
ax2.axvline(s_max, color="k", ls="--", lw=1.0)
ax2.text(s_max, 95, f"s_max={s_max:.1f}", fontsize=9, ha="right")
ax2.set_xlabel("累计行驶距离 s / m")
ax2.set_ylabel("方差占比 / %")
ax2.set_title("(b) 三项误差源相对贡献")
ax2.set_ylim(0, 100)
ax2.grid(alpha=0.3); ax2.legend(loc="center right", fontsize=8)

plt.tight_layout()
out_fig = os.path.join(FIG_DIR, "problem3.png")
plt.savefig(out_fig, dpi=150, bbox_inches="tight")
print(f"\n图已保存: {out_fig}")
print(f"结果已保存: {RES_PATH}")
plt.show()
