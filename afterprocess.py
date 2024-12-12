"""
Author: YiJin Li
Email: your.email@example.com
Date: 2024-11-28
"""

import os
import numpy as np
import tifffile as tiff

# 输入和输出目录路径
input_dir = r'E:\DP\1sam_point_updata\predict'  # 替换为图像目录路径
output_dir = r'E:\DP\1sam_point_updata\final'  # 替换为输出目录路径

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 遍历输入目录中的所有.tif文件
for filename in os.listdir(input_dir):
    if filename.endswith('.tif'):
        # 读取.tif图像
        image_path = os.path.join(input_dir, filename)
        image = tiff.imread(image_path)

        # 应用阈值：大于3赋值255，小于等于3赋值0
        threshold_value = 2.5
        binary_image = np.where(image > threshold_value, 255, 0).astype(np.uint8)

        # 保存二值化后的图像
        output_path = os.path.join(output_dir, filename)
        tiff.imwrite(output_path, binary_image)

        print(f'Processed {filename}, saved to {output_path}')
