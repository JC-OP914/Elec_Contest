# -*- coding: utf-8 -*-
"""
主入口
=====
启动摄像头、显示管线、深度学习识别，WiFi 图传。

运行环境: Canaan K230 嵌入式设备 (CanMV v1.5)
WiFi 由系统固件自动管理，不需要在代码里连接。
"""

import time, gc, sys
import socket
import network

# 确保项目目录和 SD 卡根目录都在 import 路径中
for p in ["/sdcard/Elec_Contest", "/sdcard"]:
    if p not in sys.path:
        sys.path.insert(0, p)

from machine import Pin
from libs.PipeLine import PipeLine

from detector import Detector
from streamer import MJPEGStreamer

# ==================== 硬件参数 ====================
DISPLAY_MODE = "jd9852"
RGB888P_SIZE = [1280, 720]
ROOT_PATH = "/sdcard/mp_deployment_source"
CONFIG_PATH = ROOT_PATH + "/deploy_config.json"

# ==================== 图传配置 ====================
STREAM_PORT = 80

# ==================== 初始化 ====================

print("\n===== K230 Detection Start =====")

# LCD 背光
lcd_backlight = Pin(25, Pin.OUT, pull=Pin.PULL_NONE, drive=7)
lcd_backlight.value(1)
print("[init] LCD backlight on")

# 摄像头 + 显示 pipeline
# to_ide=False: 独立运行时不需要 IDE 视频回传，避免 IDE interrupt 导致重启
print("[init] Creating pipeline...")
pipeline = PipeLine(rgb888p_size=RGB888P_SIZE, display_mode=DISPLAY_MODE)
pipeline.create(to_ide=False)
display_size = pipeline.get_display_size()
print("[init] Pipeline ready, display:", display_size)

# 深度学习检测器
print("[init] Loading detector...")
detector = Detector(CONFIG_PATH, RGB888P_SIZE, display_size)
detector.init()
print("[init] Detector ready, labels:", detector.labels)

# WiFi 图传 (WiFi 本身由系统固件自动连接)
# WiFi 连接
print("[init] Connecting WiFi...")
wlan = network.WLAN(network.STA_IF)
wlan.active(True)
if not wlan.isconnected():
    wlan.connect("B4-401", "ROBOT-B4-401")
    for _ in range(20):
        if wlan.isconnected():
            break
        time.sleep(1)
        print(".", end="")
    print()
if wlan.isconnected():
    print("[init] WiFi OK, IP:", wlan.ifconfig()[0])
else:
    print("[init] WARNING: WiFi connect failed, status:", wlan.status())

# Streamer
print("[init] Starting stream server...")
streamer = MJPEGStreamer(STREAM_PORT)
streamer.start_server()

print("[init] Entering main loop\n")

# ==================== 主循环 ====================
last_time = time.ticks_ms()
frame_count = 0
fps = 0
stable_count = 0
stable_frames = 0

try:
    while True:
        # 1. 采集画面
        img = pipeline.get_frame()

        # 2. 推理
        res = detector.infer(img)

        # 3. 绘制检测结果
        detector.draw(pipeline.osd_img, res)

        # 4. 稳定帧逻辑
        raw_count = len(res["boxes"])
        if raw_count == stable_count:
            stable_frames += 1
        else:
            stable_count = raw_count
            stable_frames = 0
        show_count = stable_count if stable_frames < 3 else raw_count

        # 5. FPS
        frame_count += 1
        now = time.ticks_ms()
        elapsed = time.ticks_diff(now, last_time)
        if elapsed >= 1000:
            fps = frame_count * 1000 // elapsed
            frame_count = 0
            last_time = now

        # 6. OSD 信息
        pipeline.osd_img.draw_string_advanced(
            5, 5, 14, "count:%d" % show_count, color=(0, 255, 0))
        pipeline.osd_img.draw_string_advanced(
            5, 22, 14, "fps:%d" % fps, color=(0, 255, 0))

        # 7. 显示到 LCD
        pipeline.show_image()

        # 8. WiFi 图传
        if streamer.accept_client():
            try:
                jpeg = pipeline.osd_img.compress(quality=70)
                if jpeg:
                    streamer.send_frame(jpeg)
            except Exception as e:
                print("[stream] JPEG error:", e)

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
    streamer.close()
    print("[main] Done")
