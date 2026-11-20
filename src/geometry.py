# -*- coding: utf-8 -*-
"""几何工具: 向量化的点到矩形距离 / 线段-长方体相交 / 可观测判定。"""
import numpy as np
from src.config import BUILDINGS, VEHICLE_RADIUS, CAMERA_HEIGHT, FIELD


def dist_to_rect(X, Y, xmin, xmax, ymin, ymax):
    """点到轴对齐矩形的最短距离 (向量化, 点在矩形内返回 0)。"""
    dx = np.maximum(np.maximum(xmin - X, X - xmax), 0.0)
    dy = np.maximum(np.maximum(ymin - Y, Y - ymax), 0.0)
    return np.hypot(dx, dy)


def _slab_axis(t_enter, t_exit, hit, P, d, lo, hi):
    """单轴 slab 更新 (向量化, 原地修改 t_enter/t_exit/hit)。"""
    d = np.asarray(d, float)
    P = np.asarray(P, float)
    parallel = np.abs(d) < 1e-12
    hit &= ~(parallel & ((P < lo) | (P > hi)))
    active = (~parallel) & hit
    if active.any():
        t1 = (lo - P[active]) / d[active]
        t2 = (hi - P[active]) / d[active]
        t_enter[active] = np.maximum(t_enter[active], np.minimum(t1, t2))
        t_exit[active] = np.minimum(t_exit[active], np.maximum(t1, t2))
        hit[active] &= t_enter[active] <= t_exit[active]


def segments_intersect_box(X, Y, gx, gy, gz, box):
    """
    三维线段 (X,Y,CAMERA_HEIGHT)->(gx,gy,gz) 是否与长方体相交 (向量化)。
    相切/接触也算遮挡 (闭区间判定)。
    """
    xmin, xmax, ymin, ymax, zmin, zmax = box
    t0 = np.zeros_like(X, float)
    t1 = np.ones_like(X, float)
    hit = np.ones_like(X, bool)
    _slab_axis(t0, t1, hit, X, gx - X, xmin, xmax)
    _slab_axis(t0, t1, hit, Y, gy - Y, ymin, ymax)
    _slab_axis(t0, t1, hit, np.full_like(X, CAMERA_HEIGHT),
               np.full_like(X, gz - CAMERA_HEIGHT), zmin, zmax)
    return hit


def observable_mask(X, Y, gx, gy, gz, elev_range=(-10.0, 80.0), elev_tol=0.0):
    """
    可观测判定 (向量化), 同时满足三条约束:
      1) 车辆圆盘不越界、不进楼;
      2) 视线仰角在云台范围内;
      3) 视线不被建筑遮挡。
    X/Y 为网格或任意数组; 返回布尔数组。
    """
    mask = np.ones(X.shape, bool)
    # 约束1
    mask &= (X >= VEHICLE_RADIUS) & (X <= FIELD[1] - VEHICLE_RADIUS)
    mask &= (Y >= VEHICLE_RADIUS) & (Y <= FIELD[3] - VEHICLE_RADIUS)
    for _, xmin, xmax, ymin, ymax, _ in BUILDINGS:
        mask &= dist_to_rect(X, Y, xmin, xmax, ymin, ymax) >= VEHICLE_RADIUS
    # 约束2
    r_h = np.hypot(gx - X, gy - Y)
    elev = np.degrees(np.arctan2(gz - CAMERA_HEIGHT, r_h))
    mask &= (elev >= elev_range[0] - elev_tol) & (elev <= elev_range[1] + elev_tol)
    # 约束3
    for _, xmin, xmax, ymin, ymax, h in BUILDINGS:
        mask &= ~segments_intersect_box(X, Y, gx, gy, gz,
                                        (xmin, xmax, ymin, ymax, 0.0, h))
    return mask


def elevation_deg(r_h, gz):
    """由水平距离 r 求视线仰角 (度)。"""
    return np.degrees(np.arctan2(gz - CAMERA_HEIGHT, r_h))
