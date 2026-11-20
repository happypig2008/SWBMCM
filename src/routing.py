# -*- coding: utf-8 -*-
"""绕楼最短路径: 网格 Dijkstra (8 邻接)。"""
import heapq
import numpy as np
from src.config import FIELD, BUILDINGS, VEHICLE_RADIUS
from src.geometry import dist_to_rect


def make_grid(step):
    """构建场地网格与障碍掩码。
    返回 (xs, ys, obstacle): obstacle[j, i] = True 表示该格点不可通行。
    """
    xs = np.arange(FIELD[0], FIELD[1] + step, step)
    ys = np.arange(FIELD[2], FIELD[3] + step, step)
    Xg, Yg = np.meshgrid(xs, ys)
    obstacle = np.zeros(Xg.shape, dtype=bool)
    for _, xmin, xmax, ymin, ymax, _ in BUILDINGS:
        obstacle |= dist_to_rect(Xg, Yg, xmin, xmax, ymin, ymax) < VEHICLE_RADIUS
    return xs, ys, obstacle


def shortest_path_distances(start, xs, ys, obstacle):
    """从 start(x, y) 到所有自由格点的最短路径距离 (Dijkstra, 8 邻接)。
    返回 dist: 形状同 obstacle, 单位米; 障碍/不可达处为 inf。
    """
    step = float(xs[1] - xs[0])
    ny, nx = obstacle.shape
    si = int(round((start[0] - xs[0]) / step))
    sj = int(round((start[1] - ys[0]) / step))
    si = min(max(si, 0), nx - 1)
    sj = min(max(sj, 0), ny - 1)

    dist = np.full((ny, nx), np.inf)
    dist[sj, si] = 0.0
    # 8 邻接: (dx, dy, 代价系数)
    neighbors = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                 (-1, -1, np.sqrt(2)), (-1, 1, np.sqrt(2)),
                 (1, -1, np.sqrt(2)), (1, 1, np.sqrt(2))]
    pq = [(0.0, sj, si)]
    while pq:
        d, j, i = heapq.heappop(pq)
        if d > dist[j, i]:
            continue
        for di, dj, cm in neighbors:
            ii = i + di
            jj = j + dj
            if 0 <= ii < nx and 0 <= jj < ny and not obstacle[jj, ii]:
                nd = d + step * cm
                if nd < dist[jj, ii]:
                    dist[jj, ii] = nd
                    heapq.heappush(pq, (nd, jj, ii))
    return dist
