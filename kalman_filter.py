# -*- coding: utf-8 -*-
"""
EMA + 速度预测 检测框平滑器
===========================
1. 估计速度:  v = pos - prev_pos
2. 预测位置:  pred = pos + v (抵消 EMA 滞后)
3. EMA 平滑:  out = alpha * meas + (1-alpha) * pred
"""

class KalmanBoxFilter:
    def __init__(self, alpha=0.65):
        self.alpha = alpha
        self.beta = 1.0 - alpha
        self._init = False
        self.x = 0.0; self.y = 0.0
        self.w = 0.0; self.h = 0.0
        self.vx = 0.0; self.vy = 0.0

    def predict(self):
        """用速度预测下一帧位置"""
        if self._init:
            self.x += self.vx
            self.y += self.vy

    def update(self, box):
        """用测量值更新"""
        if not self._init:
            self.x, self.y = box[0], box[1]
            self.w, self.h = box[2], box[3]
            self.vx = self.vy = 0.0
            self._init = True
        else:
            # 估计速度
            new_vx = box[0] - self.x
            new_vy = box[1] - self.y
            self.vx = 0.7 * new_vx + 0.3 * self.vx
            self.vy = 0.7 * new_vy + 0.3 * self.vy

            # EMA 平滑
            self.x = self.alpha * box[0] + self.beta * self.x
            self.y = self.alpha * box[1] + self.beta * self.y
            self.w = self.alpha * box[2] + self.beta * self.w
            self.h = self.alpha * box[3] + self.beta * self.h

    def get_box(self):
        return [int(self.x), int(self.y),
                max(1, int(self.w)), max(1, int(self.h))]

    def reset(self):
        self._init = False
