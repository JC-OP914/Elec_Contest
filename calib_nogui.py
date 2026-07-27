"""
无GUI版本的相机内参标定脚本
基于 intrinsicCalib.py 修改，移除所有 GUI 相关代码
"""

import cv2
import numpy as np
import os
import argparse

# 参数配置
INPUT_PATH = './data/'
IMAGE_FILE = 'img_raw'
CAMERA_TYPE = 'normal'
FRAME_WIDTH = 320
FRAME_HEIGHT = 240
BORAD_WIDTH = 9   # 内角点宽度
BORAD_HEIGHT = 6  # 内角点高度
SQUARE_SIZE = 11  # 方格边长(mm)
CALIB_NUMBER = 30 # 最少需要的有效标定帧数
SUBPIX_REGION = 5

class CalibData:
    def __init__(self):
        self.type = None
        self.camera_mat = None
        self.dist_coeff = None
        self.rvecs = None
        self.tvecs = None
        self.reproj_err = None
        self.ok = False

class Normal:
    def __init__(self):
        self.data = CalibData()
        self.inited = False
        self.BOARD = np.array([ [(j * SQUARE_SIZE, i * SQUARE_SIZE, 0.)]
                               for i in range(BORAD_HEIGHT) 
                               for j in range(BORAD_WIDTH) ],dtype=np.float32)
        
    def update(self, corners, frame_size):
        board = [self.BOARD] * len(corners)
        if not self.inited:
            self._update_init(board, corners, frame_size)
            self.inited = True
        else:
            self._update_refine(board, corners, frame_size)
        self._calc_reproj_err(corners)

    def _update_init(self, board, corners, frame_size):
        data = self.data
        data.type = "NORMAL"
        data.camera_mat = np.eye(3, 3)
        data.dist_coeff = np.zeros((5, 1))
        data.ok, data.camera_mat, data.dist_coeff, data.rvecs, data.tvecs = cv2.calibrateCamera(
            board, corners, frame_size, data.camera_mat, data.dist_coeff, 
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 1e-6))
        data.ok = data.ok and cv2.checkRange(data.camera_mat) and cv2.checkRange(data.dist_coeff)
        
    def _update_refine(self, board, corners, frame_size):
        data = self.data
        data.ok, data.camera_mat, data.dist_coeff, data.rvecs, data.tvecs = cv2.calibrateCamera(
            board, corners, frame_size, data.camera_mat, data.dist_coeff,  
            flags = cv2.CALIB_USE_INTRINSIC_GUESS,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 10, 1e-6))
        data.ok = data.ok and cv2.checkRange(data.camera_mat) and cv2.checkRange(data.dist_coeff)
        
    def _calc_reproj_err(self, corners):
        if not self.inited: return
        data = self.data
        data.reproj_err = []
        for i in range(len(corners)):
            corners_reproj, _ = cv2.projectPoints(self.BOARD, data.rvecs[i], data.tvecs[i], data.camera_mat, data.dist_coeff)
            err = cv2.norm(corners_reproj, corners[i], cv2.NORM_L2) / len(corners_reproj)
            data.reproj_err.append(err)


def get_images(PATH, NAME):
    filePath = [os.path.join(PATH, x) for x in os.listdir(PATH) 
                if any(x.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])
               ]
    filenames = [filename for filename in filePath if NAME in filename]
    if len(filenames) == 0:
        raise Exception("from {} read images failed".format(PATH))
    return filenames


def main():
    print("=" * 60)
    print("相机内参标定 (无GUI模式)")
    print("=" * 60)
    print(f"输入路径: {INPUT_PATH}")
    print(f"图片前缀: {IMAGE_FILE}")
    print(f"相机类型: {CAMERA_TYPE}")
    print(f"图像尺寸: {FRAME_WIDTH}x{FRAME_HEIGHT}")
    print(f"棋盘格角点: {BORAD_WIDTH}x{BORAD_HEIGHT}")
    print(f"方格边长: {SQUARE_SIZE}mm")
    print(f"最少需要: {CALIB_NUMBER}张有效图片")
    print("=" * 60)
    
    camera = Normal()
    corners_list = []
    
    # 获取所有图片
    filenames = get_images(INPUT_PATH, IMAGE_FILE)
    print(f"\n找到 {len(filenames)} 张图片\n")
    
    # 处理每张图片
    valid_count = 0
    for i, filename in enumerate(sorted(filenames)):
        img = cv2.imread(filename)
        if img is None:
            print(f"[{i+1}/{len(filenames)}] 无法读取: {os.path.basename(filename)}")
            continue
            
        # 缩放到指定尺寸
        img_resized = cv2.resize(img, (FRAME_WIDTH, FRAME_HEIGHT))
        
        # 检测棋盘格角点
        ok, corners = cv2.findChessboardCorners(img_resized, (BORAD_WIDTH, BORAD_HEIGHT),
                      flags=cv2.CALIB_CB_ADAPTIVE_THRESH|cv2.CALIB_CB_NORMALIZE_IMAGE|cv2.CALIB_CB_FAST_CHECK)
        
        if ok:
            gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            corners = cv2.cornerSubPix(gray, corners, (SUBPIX_REGION, SUBPIX_REGION), (-1, -1),
                                       (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
            corners_list.append(corners)
            valid_count += 1
            print(f"[{i+1}/{len(filenames)}] ✓ 检测成功: {os.path.basename(filename)} (累计: {valid_count})")
        else:
            print(f"[{i+1}/{len(filenames)}] ✗ 未检测到棋盘格: {os.path.basename(filename)}")
    
    print(f"\n{'=' * 60}")
    print(f"检测完成! 有效图片: {valid_count}/{len(filenames)}")
    
    if valid_count == 0:
        raise Exception("标定失败! 未检测到任何有效的棋盘格图片，请检查参数设置")
    
    if valid_count < CALIB_NUMBER:
        print(f"警告: 有效图片数量({valid_count})少于要求的最少数量({CALIB_NUMBER})")
        print("标定结果可能不准确，建议增加更多标定图片")
    
    # 执行标定
    print("\n正在执行标定...")
    camera.update(corners_list, (FRAME_WIDTH, FRAME_HEIGHT))
    
    result = camera.data
    
    if not result.ok:
        raise Exception("标定失败! 请检查输入数据")
    
    # 输出结果
    print("\n" + "=" * 60)
    print("标定完成!")
    print("=" * 60)
    print(f"\n相机类型: {result.type}")
    print(f"\n相机内参矩阵 K:")
    print(result.camera_mat)
    print(f"\n畸变系数 D:")
    print(result.dist_coeff.flatten())
    print(f"\n重投影误差: {np.mean(result.reproj_err):.4f} 像素")
    
    # 保存结果
    np.save('camera_1_K.npy', result.camera_mat.tolist())
    np.save('camera_1_D.npy', result.dist_coeff.tolist())
    print(f"\n结果已保存至:")
    print(f"  - camera_1_K.npy (内参矩阵)")
    print(f"  - camera_1_D.npy (畸变系数)")
    print("=" * 60)


if __name__ == '__main__':
    main()
