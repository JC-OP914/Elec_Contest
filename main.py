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
from kalman_filter import KalmanBoxFilter
from uart_comm import UartComm

# ==================== 配置 ====================
DISPLAY_MODE = "jd9852"
RGB888P_SIZE = [1280, 720]
ROOT_PATH = "/sdcard/mp_deployment_source"
CONFIG_PATH = ROOT_PATH + "/deploy_config.json"

X_OFFSET_CM = 0.8      # zero-point compensation (cm), + = shift right
CX_OFFSET_PX = 4       # dashed line offset on display (px), + = right

# ROI: full-width horizontal strip centered vertically
ROI_HEIGHT = 100       # ROI height (px), tune based on pipe diameter ~3cm
ROI = [0, (RGB888P_SIZE[1] - ROI_HEIGHT) // 2, RGB888P_SIZE[0], ROI_HEIGHT]
# ROI = None           # uncomment to disable ROI, use full frame

# ==================== 相机标定 (1280x720) ====================
CAMERA_MATRIX = [
    1067.38698804, 0.0, 613.30479540,
    0.0, 1062.97875192, 396.85559676,
    0.0, 0.0, 1.0
]
DIST_COEFFS = [0.05526456, -0.4621188, 0.00494004, -0.00092707, 0.09250291]
IMAGE_SHAPE = [RGB888P_SIZE[1], RGB888P_SIZE[0]]  # [720, 1280]

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
SX = display_size[0] / RGB888P_SIZE[0]
SY = display_size[1] / RGB888P_SIZE[1]
print("[init] Coord scale: SX=%.3f SY=%.3f" % (SX, SY))

print("[init] Loading detector...")
detector = Detector(CONFIG_PATH, RGB888P_SIZE, display_size, roi=ROI)
detector.init()
print("[init] Detector ready, labels:", detector.labels)
if ROI:
    print("[init] ROI:", ROI)

print("[init] Loading PnP estimator...")
pnp = PnPEstimator(IMAGE_SHAPE, CAMERA_MATRIX, DIST_COEFFS,
                   ball_diameter_cm=1.0, x_offset_cm=X_OFFSET_CM)
print("[init] PnP ready")

print("[init] Init Kalman filter...")
kf = KalmanBoxFilter()
print("[init] Kalman ready")

print("[init] Init UART...")
uart = UartComm(uart_id=3, baudrate=115200,
                tx_pin=50, rx_pin=51, send_interval_ms=50)
print("[init] UART ready")

print("[init] Entering main loop\n")

# ==================== 主循环 ====================
last_time = time.ticks_ms()
frame_count = 0
fps = 0
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
last_raw_box = None     # hold last raw detection box
lost_frames = 0
last_score = 0.0
MAX_LOST = 15

try:
    while True:
        img = pipeline.get_frame()
        pipeline.osd_img.clear()

        # ---- draw ROI border (blue) ----
        if ROI:
            rx_d = int(ROI[0] * SX)
            ry_d = int(ROI[1] * SY)
            rw_d = int(ROI[2] * SX)
            rh_d = int(ROI[3] * SY)
            pipeline.osd_img.draw_rectangle(
                rx_d, ry_d, rw_d, rh_d, color=BLUE, thickness=2)

        res = detector.infer(img)

        boxes = res.get("boxes", [])
        raw_count = len(boxes)

        # ---- Kalman ----
        kf.predict()
        ball_pos = None
        kf_box = None
        raw_show = False

        if raw_count > 0:
            box = boxes[0]
            box_raw = [box[0], box[1], box[2] - box[0], box[3] - box[1]]
            last_raw_box = box
            lost_frames = 0
            last_score = res["scores"][0] if len(res["scores"]) > 0 else 0.0
            kf.update(box_raw)
        else:
            lost_frames += 1
            if lost_frames > MAX_LOST:
                kf.reset()

        raw_show = (last_raw_box is not None and lost_frames <= MAX_LOST)

        # ---- draw raw detection: red box + red cross (held) ----
        if raw_show:
            lb = last_raw_box
            x1 = int(lb[0] * SX)
            y1 = int(lb[1] * SY)
            x2 = int(lb[2] * SX)
            y2 = int(lb[3] * SY)
            pipeline.osd_img.draw_rectangle(
                x1, y1, x2 - x1, y2 - y1, color=RED, thickness=2)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            pipeline.osd_img.draw_string_advanced(
                cx - 4, cy - 7, 14, "+", color=RED)
            r = 5
            pipeline.osd_img.draw_rectangle(
                cx - r, cy - 1, r * 2, 2, color=RED, thickness=-1)
            pipeline.osd_img.draw_rectangle(
                cx - 1, cy - r, 2, r * 2, color=RED, thickness=-1)

        # ---- draw kalman filtered: green box + green cross ----
        if kf._init:
            kf_box = kf.get_box()
            bx_s, by_s, bw_s, bh_s = kf_box
            bx = int(bx_s * SX)
            by = int(by_s * SY)
            bw = int(bw_s * SX)
            bh = int(bh_s * SY)
            pipeline.osd_img.draw_rectangle(
                bx, by, bw, bh, color=GREEN, thickness=2)
            pipeline.osd_img.draw_string_advanced(
                bx + bw // 2 - 4, by + bh // 2 - 7, 14, "+", color=GREEN)
            # PnP from kalman box
            if pipeline.cur_frame:
                ball_pos = pnp.estimate(img, kf_box)
                if ball_pos:
                    uart.send(ball_pos[0])

        # ---- FPS ----
        frame_count += 1
        now = time.ticks_ms()
        elapsed = time.ticks_diff(now, last_time)
        if elapsed >= 1000:
            fps = frame_count * 1000 // elapsed
            frame_count = 0
            last_time = now
            if ball_pos:
                print("[PnP] x=%.2f cm  z=%.2f cm  fps=%d" % (ball_pos[0], ball_pos[1], fps))

        # ---- left OSD ----
        show_count = 1 if (raw_show or kf._init) else 0
        pipeline.osd_img.draw_string_advanced(
            5, 5, 14, "count:%d" % show_count, color=(0, 255, 0))
        pipeline.osd_img.draw_string_advanced(
            5, 22, 14, "fps:%d" % fps, color=(0, 255, 0))
        if ball_pos:
            x_cm, z_cm = ball_pos
            pipeline.osd_img.draw_string_advanced(
                5, 39, 14, "x:%.1f cm" % x_cm, color=(255, 255, 0))
            pipeline.osd_img.draw_string_advanced(
                5, 56, 14, "dist:%.1f cm" % z_cm, color=(255, 255, 0))

        # ---- top-right: confidence ----
        if raw_show:
            conf_text = "conf:%.2f" % last_score
            conf_x = display_size[0] - len(conf_text) * 9 - 10
            pipeline.osd_img.draw_string_advanced(
                conf_x, 5, 14, conf_text, color=(255, 255, 0))

        # ---- center dashed line ----
        # dashed line at zero-point (fixed, camera won't move)
        cx_disp = display_size[0] // 2 + CX_OFFSET_PX
        for dy in range(0, display_size[1], 16):
            pipeline.osd_img.draw_string_advanced(
                cx_disp - 1, dy, 12, "|", color=(0, 0, 255))

        pipeline.show_image()
        gc.collect()

except KeyboardInterrupt:
    print("\n[main] User interrupted")
except Exception as e:
    print("[main] Fatal error:", e)
    sys.print_exception(e)
finally:
    print("[main] Cleaning up...")
    uart.deinit()
    detector.deinit()
    pipeline.destroy()
    print("[main] Done")
