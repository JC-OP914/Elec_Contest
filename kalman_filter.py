# -*- coding: utf-8 -*-
"""
指数滑动平均 (EMA) 检测框平滑器
================================
每个坐标独立平滑:  new = alpha * meas + (1-alpha) * old
alpha 越大越灵敏, 越小越平滑
"""

class KalmanBoxFilter:
    def __init__(self, alpha=0.65):
        self.alpha = alpha
        self.beta = 1.0 - alpha
        self._init = False
        self.x = 0.0; self.y = 0.0
        self.w = 0.0; self.h = 0.0

    def predict(self):
        pass  # EMA 不需要预测, 保留接口兼容

    def update(self, box):
        if not self._init:
            self.x, self.y = box[0], box[1]
            self.w, self.h = box[2], box[3]
            self._init = True
        else:
            self.x = self.alpha * box[0] + self.beta * self.x
            self.y = self.alpha * box[1] + self.beta * self.y
            self.w = self.alpha * box[2] + self.beta * self.w
            self.h = self.alpha * box[3] + self.beta * self.h

    def get_box(self):
        return [int(self.x), int(self.y),
                max(1, int(self.w)), max(1, int(self.h))]

    def reset(self):
        self._init = False
