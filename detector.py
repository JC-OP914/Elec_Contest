# -*- coding: utf-8 -*-
"""
深度学习识别模块
===============
封装 DetectionApp，负责模型加载和推理。
"""

from libs.PlatTasks import DetectionApp
from libs.Utils import read_json


class Detector:
    def __init__(self, config_path: str, rgb888p_size: list, display_size: list,
                 inference_mode: str = "video", debug_mode: int = 0):
        """
        :param config_path: deploy_config.json 路径
        :param rgb888p_size: 视频帧尺寸 [w, h]
        :param display_size: 显示尺寸 [w, h] (由 Pipeline.get_display_size() 获取)
        :param inference_mode: 'video' 或 'image'
        :param debug_mode: 0=关闭, 1=开启
        """
        conf = read_json(config_path)
        root = config_path.rsplit("/", 1)[0]

        self.labels = conf["categories"]
        self.kmodel_path = root + "/" + conf["kmodel_path"]
        self.model_input_size = conf["img_size"]
        self.confidence_threshold = conf["confidence_threshold"]
        self.nms_threshold = conf["nms_threshold"]
        self.nms_option = conf["nms_option"]
        self.model_type = conf["model_type"]

        # 处理 anchors (AnchorBaseDet 需要)
        self.anchors = []
        if self.model_type == "AnchorBaseDet":
            raw = conf["anchors"]
            self.anchors = raw[0] + raw[1] + raw[2]

        self.inference_mode = inference_mode
        self.debug_mode = debug_mode
        self.rgb888p_size = rgb888p_size
        self.display_size = display_size

        self._app = None

    def init(self):
        """初始化 DetectionApp 并配置预处理"""
        self._app = DetectionApp(
            self.inference_mode,
            self.kmodel_path,
            self.labels,
            self.model_input_size,
            self.anchors,
            self.model_type,
            self.confidence_threshold,
            self.nms_threshold,
            self.rgb888p_size,
            self.display_size,
            debug_mode=self.debug_mode,
        )
        self._app.config_preprocess()

    def infer(self, img) -> dict:
        """
        单帧推理
        :param img: 输入图像 (来自 pipeline.get_frame())
        :return: {"boxes": [...], "classes": [...], ...}
        """
        return self._app.run(img)

    def draw(self, osd_img, result: dict):
        """在 OSD 图像上绘制检测结果"""
        self._app.draw_result(osd_img, result)

    def deinit(self):
        """释放资源"""
        if self._app:
            self._app.deinit()
            self._app = None
