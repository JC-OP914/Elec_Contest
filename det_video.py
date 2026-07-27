# -*- coding: utf-8 -*-
'''
Script: deploy_det_video.py
脚本名称：deploy_det_video.py

Description:
    This script runs a real-time object detection application on an embedded device.
    It uses a pipeline to capture video frames, performs inference using a pre-trained Kmodel,
    and displays the detection results (bounding boxes, class labels) on screen.

    The model configuration is loaded from the Canaan online training platform via a JSON config file.

脚本说明：
    本脚本在嵌入式设备上运行实时目标检测应用。它通过捕获视频帧，使用预训练的 Kmodel 进行推理，并在屏幕上显示检测结果（边界框、类别标签）。

    模型配置文件通过 Canaan 在线训练平台从 JSON 文件加载。

Author: Canaan Developer
作者：Canaan 开发者
'''


import os, gc
from machine import Pin
from libs.PlatTasks import DetectionApp
from libs.PipeLine import PipeLine
from libs.Utils import *

# Set display mode: options are 'hdmi', 'lcd', 'lt9611', 'st7701', 'hx8399', 'jd9852'
# 'hdmi' defaults to 'lt9611' (1920x1080); 'lcd' defaults to 'st7701' (800x480)
display_mode = "jd9852"

# WonderMK LCD backlight (GPIO 25)
lcd_backlight = Pin(25, Pin.OUT, pull=Pin.PULL_NONE, drive=7)
lcd_backlight.value(1)

# Define the input size for the RGB888P video frames
rgb888p_size = [1280, 720]

# Set root directory path for model and config
root_path = "/sdcard/mp_deployment_source"

# Load deployment configuration
deploy_conf = read_json(root_path + "/deploy_config.json")
kmodel_path = root_path + "/" + deploy_conf["kmodel_path"]         # KModel path
labels = deploy_conf["categories"]                                # Label list
confidence_threshold = deploy_conf["confidence_threshold"]        # Confidence threshold
nms_threshold = deploy_conf["nms_threshold"]                      # NMS threshold
model_input_size = deploy_conf["img_size"]                        # Model input size
nms_option = deploy_conf["nms_option"]                            # NMS strategy
model_type = deploy_conf["model_type"]                            # Detection model type
anchors = []
if model_type == "AnchorBaseDet":
    anchors = deploy_conf["anchors"][0] + deploy_conf["anchors"][1] + deploy_conf["anchors"][2]

# Inference configuration
inference_mode = "video"                                          # Inference mode: 'video'
debug_mode = 0                                                    # Debug mode flag

# Create and initialize the video/display pipeline
pl = PipeLine(rgb888p_size=rgb888p_size, display_mode=display_mode)
pl.create(to_ide=True)
display_size = pl.get_display_size()

# Initialize object detection application
det_app = DetectionApp(inference_mode,kmodel_path,labels,model_input_size,anchors,model_type,confidence_threshold,nms_threshold,rgb888p_size,display_size,debug_mode=debug_mode)

# Configure preprocessing for the model
det_app.config_preprocess()

# Main loop: capture, run inference, display results
last_time = time.ticks_ms()
frame_count = 0
fps = 0
stable_count = 0
stable_frames = 0
while True:
    img = pl.get_frame()                          # Capture current frame
    res = det_app.run(img)                        # Run inference
    det_app.draw_result(pl.osd_img, res)          # Draw detection results
    raw_count = len(res["boxes"])
    if raw_count == stable_count:
        stable_frames += 1
    else:
        stable_count = raw_count
        stable_frames = 0
    if stable_frames < 3:
        raw_count = stable_count
    frame_count += 1
    now = time.ticks_ms()
    elapsed = time.ticks_diff(now, last_time)
    if elapsed >= 1000:
        fps = frame_count * 1000 // elapsed
        frame_count = 0
        last_time = now
    pl.osd_img.draw_string_advanced(5, 5, 14, "count:%d" % raw_count, color=(0, 255, 0))
    pl.osd_img.draw_string_advanced(5, 22, 14, "fps:%d" % fps, color=(0, 255, 0))
    pl.show_image()                               # Show result on display
    gc.collect()                                  # Run garbage collection

# Cleanup: These lines will only run if the loop is interrupted (e.g., by an IDE break or external interruption)
det_app.deinit()                                      # De-initialize detection app
pl.destroy()                                          # Destroy pipeline instance
