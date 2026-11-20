# -*- coding: utf-8 -*-
"""绘图公共工具: 中文字体 + 建筑绘制 + 保存。"""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from src.config import BUILDINGS


def setup_style():
    """全局中文字体与负号配置。"""
    matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "SimSun"]
    matplotlib.rcParams["axes.unicode_minus"] = False


def draw_buildings(ax, label_height=True):
    """在 ax 上绘制所有建筑 (灰色矩形)。"""
    for name, xmin, xmax, ymin, ymax, h in BUILDINGS:
        ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                               facecolor="gray", edgecolor="k", lw=1.2))
        if label_height:
            ax.text((xmin + xmax) / 2, (ymin + ymax) / 2, f"{name}\n{h}m",
                    ha="center", va="center", fontsize=8, color="white")


def save_fig(fig, path, dpi=150):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
