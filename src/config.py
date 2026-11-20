# -*- coding: utf-8 -*-
"""全局配置: 场地、建筑、车辆、云台、目标。各问题 (task1~task4) 复用。"""
import numpy as np

FIELD = (0.0, 100.0, 0.0, 100.0)   # (xmin, xmax, ymin, ymax)
DISPATCH_CENTER = (50.0, 50.0)     # 调度中心 O

VEHICLE_RADIUS = 1.5               # 车辆等效圆盘半径 m
VEHICLE_SPEED = 3.0                # 行驶速度 m/s
CAMERA_HEIGHT = 2.0                # 相机离地高度 m
GIMBAL_ELEV = (-10.0, 80.0)        # 云台俯仰角范围 (度)
GIMBAL_ANGULAR_SPEED = 30.0        # 云台两轴最大角速度 度/s
JOINT_OBS_TIME = 5.0               # 联合观测时长 s

# 建筑: (名称, xmin, xmax, ymin, ymax, 高度)
BUILDINGS = [
    ("B1", 22, 34, 16, 26, 20),
    ("B2", 63, 79, 59, 71, 28),
    ("B3", 10, 24, 65, 76, 12),
    ("B4", 72, 86, 22, 34, 24),
    ("B5", 41, 51, 38, 46, 32),
]

# 车辆初始位置
VEHICLES = {
    "V1": (15, 14), "V2": (88, 10), "V3": (14, 86),
    "V4": (88, 68), "V5": (50, 93),
}

# 警报目标: (方位角°, 水平距离 m, 高度 m)
ALERTS = {
    "T1": (30.0, 40.0, 45.0),
    "T2": (150.0, 45.0, 35.0),
    "T3": (285.0, 38.0, 55.0),
}


def alert_xyz(name, center=DISPATCH_CENTER):
    """由警报名求目标三维坐标 (x, y, z)。方位角从正东逆时针量。"""
    theta, R, H = ALERTS[name]
    t = np.radians(theta)
    return (center[0] + R * np.cos(t), center[1] + R * np.sin(t), H)


# ---- 可调参数 (各问题复用) ----
ROUTING_STEP = 1.0     # 绕楼最短路径网格步长 m
STATION_STEP = 1.0     # 候选观测站网格步长 m
REFINE_STEP = 0.1      # 局部精化步长 m
R_CUT = 20.0           # 候选站搜索半径(目标周围) m
T_MAX = 30.0           # 单站行驶时间上限 s
SIGMA_AZ_DEG = 1.0     # 方位角测量误差标准差 (度)
