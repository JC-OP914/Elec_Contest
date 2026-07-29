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

# ==================== 配置 ====================
DISPLAY_MODE = "jd9852"
RGB888P_SIZE = [1280, 720]
ROOT_PATH = "/sdcard/mp_deployment_source"
CONFIG_PATH = ROOT_PATH + "/deploy_config.json"

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
