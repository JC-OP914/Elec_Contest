# -*- coding: utf-8 -*-
"""
主入口
=====
摄像头 + 显示 + 深度学习识别。
运行环境: Canaan K230 (CanMV v1.5)
"""

import time, gc, sys

for p in ["/sdcard/Elec_Contest", "/sdcard"]:
    if p not in sys.path:
        sys.path.insert(0, p)

from machine import Pin
from libs.PipeLine import PipeLine
from detector import Detector
from pnp_estimator import PnPEstimator

# ==================== 配置 ====================
DISPLAY_MODE = "jd9852"
RGB888P_SIZE = [320, 240]
ROOT_PATH = "/sdcard/mp_deployment_source"
CONFIG_PATH = ROOT_PATH + "/deploy_config.json"

# ==================== 相机标定 (320x240) ====================
CAMERA_MATRIX = [
    266.84674701, 0.0, 153.32619885,
    0.0, 354.32625064, 132.28519892,
    0.0, 0.0, 1.0
]
DIST_COEFFS = [0.05526456, -0.4621188, 0.00494004, -0.00092707, 0.09250291]
IMAGE_SHAPE = [RGB888P_SIZE[1], RGB888P_SIZE[0]]  # [240, 320]

# ==================== 初始化 ====================

print("\n===== K230 Detection Start =====")

lcd_backlight = Pin(25, Pin.OUT, pull=Pin.PULL_NONE, drive=7)
lcd_backlight.value(1)
print("[init] LCD backlight on")

print("[init] Creating pipeline...")
pipeline = PipeLine(rgb888p_size=RGB888P_SIZE, display_mode=DISPLAY_MODE)
pipeline.create(to_ide=False)
display_size = pipeline.get_display_size()
print("[init] Pipeline ready, display:", display_size)

print("[init] Loading detector...")
detector = Detector(CONFIG_PATH, RGB888P_SIZE, display_size)
detector.init()
print("[init] Detector ready, labels:", detector.labels)

print("[init] Loading PnP estimator...")
pnp = PnPEstimator(IMAGE_SHAPE, CAMERA_MATRIX, DIST_COEFFS, ball_diameter_cm=1.0)
print("[init] PnP ready")

print("[init] Entering main loop\n")

# ==================== 主循环 ====================
last_time = time.ticks_ms()
frame_count = 0
fps = 0
stable_count = 0
stable_frames = 0

try:
    while True:
        img = pipeline.get_frame()
        res = detector.infer(img)
        detector.draw(pipeline.osd_img, res)

        raw_count = len(res["boxes"])
        if raw_count == stable_count:
            stable_frames += 1
        else:
            stable_count = raw_count
            stable_frames = 0
        show_count = stable_count if stable_frames < 3 else raw_count

        # PnP 位置估计
        ball_pos = None
        if raw_count > 0 and pipeline.cur_frame:
            box = res["boxes"][0]  # 取第一个钢球
            print("[main] raw box:", box)
            ball_pos = pnp.estimate(img, box)

        frame_count += 1
        now = time.ticks_ms()
        elapsed = time.ticks_diff(now, last_time)
        if elapsed >= 1000:
            fps = frame_count * 1000 // elapsed
            frame_count = 0
            last_time = now

        pipeline.osd_img.draw_string_advanced(
            5, 5, 14, "count:%d" % show_count, color=(0, 255, 0))
        pipeline.osd_img.draw_string_advanced(
            5, 22, 14, "fps:%d" % fps, color=(0, 255, 0))
        if ball_pos:
            x_cm, z_cm = ball_pos
            print("[PnP] x=%.2f cm  z=%.2f cm" % (x_cm, z_cm))
            pipeline.osd_img.draw_string_advanced(
                5, 39, 14, "pos:%.1f cm" % x_cm, color=(255, 255, 0))
            pipeline.osd_img.draw_string_advanced(
                5, 56, 14, "dist:%.1f cm" % z_cm, color=(255, 255, 0))

        pipeline.show_image()
        gc.collect()

except KeyboardInterrupt:
    print("\n[main] User interrupted")
except Exception as e:
    print("[main] Fatal error:", e)
    sys.print_exception(e)
finally:
    print("[main] Cleaning up...")
    detector.deinit()
    pipeline.destroy()
    print("[main] Done")
