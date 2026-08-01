# -*- coding: utf-8 -*-
"""
PnP 位置估计模块
===============
基于 cv_lite.rgb888_pnp_distance 官方封装，
估算钢球在摆杆上的物理位置(cm)。
"""

import cv_lite
import ulab.numpy as np


class PnPEstimator:
    def __init__(self, image_shape, camera_matrix, dist_coeffs,
                 ball_diameter_cm=1.0, x_offset_cm=0.0):
        """
        :param image_shape: [height, width] 图像尺寸
        :param camera_matrix: 3x3 内参矩阵，一维 list (9个元素)
        :param dist_coeffs: 畸变系数 list (5个元素)
        :param ball_diameter_cm: 钢球实际直径(cm)，默认 1.0
        :param x_offset_cm: x方向零点补偿(cm)，正值=零点右移
        """
        self.image_shape = image_shape
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.dist_len = len(dist_coeffs)
        self.ball_w = ball_diameter_cm
        self.ball_h = ball_diameter_cm
        self.x_offset_cm = x_offset_cm

        # 预取内参
        self.fx = camera_matrix[0]
        self.fy = camera_matrix[4]
        self.cx = camera_matrix[2]
        self.cy = camera_matrix[5]

    def estimate(self, img_np, roi):
        """
        估算钢球 3D 位置 (沿摆杆方向 x_cm, 垂直距离 z_cm)
        :param img_np: numpy 数组 (RGB888 图像数据)
        :param roi: [x, y, w, h] 检测框
        :return: (x_cm, z_cm) 或 None
        """
        x, y, w, h = roi[0], roi[1], roi[2], roi[3]
        if w <= 0 or h <= 0:
            return None

        # 边界检查
        ih, iw = self.image_shape[0], self.image_shape[1]
        if x < 0 or y < 0 or x + w > iw or y + h > ih:
            print("[pnp] ROI out of bounds: roi=%s img=%dx%d" % (roi, iw, ih))
            x, y, w, h = max(0, x), max(0, y), min(w, iw - x), min(h, ih - y)
            if w <= 0 or h <= 0:
                return None

        try:
            dist = cv_lite.rgb888_pnp_distance(
                self.image_shape, img_np, [x, y, w, h],
                self.camera_matrix, self.dist_coeffs, self.dist_len,
                self.ball_w, self.ball_h
            )
            if dist is None or dist <= 0:
                return None

            u = x + w / 2.0
            v = y + h / 2.0
            # x=0 at image center, minus offset compensation
            x_cm = (u - iw / 2.0) * dist / self.fx - self.x_offset_cm
            y_cm = (v - self.cy) * dist / self.fy
            return (x_cm, dist)

        except Exception as e:
            print("[pnp] Error:", e)
            return None
