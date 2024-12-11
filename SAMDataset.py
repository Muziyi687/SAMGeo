"""
Author: YiJin Li
Email: your.email@example.com
Date: 2024-11-28
"""

import os
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2

class SAMCompatibleDataset(Dataset):
    def __init__(self, image_dir, mask_dir, points_dir, target_size=(1024, 1024)):
        """
        初始化数据集
        :param image_dir: 图像目录
        :param mask_dir: 掩码目录
        :param points_dir: 点提示目录
        :param target_size: 目标图像尺寸
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.points_dir = points_dir
        self.target_size = target_size 

        self.image_files = sorted(os.listdir(image_dir))
        self.mask_files = sorted(os.listdir(mask_dir))
        self.points_files = sorted(os.listdir(points_dir))

        # 确保数据文件数量一致
        assert len(self.image_files) == len(self.mask_files) == len(self.points_files), \
            "图像、掩码和点文件数量不一致，请检查文件夹内容！"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        获取指定索引的数据
        :param idx: 数据索引
        :return: 图像、掩码、点提示、标签
        """
        # 加载图像
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为 RGB 格式

        # Resize 图像
        image = cv2.resize(image, self.target_size)

        # 加载掩码
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Resize 掩码
        mask = cv2.resize(mask, self.target_size)

        # 加载点提示
        points_path = os.path.join(self.points_dir, self.points_files[idx])
        points_data = np.load(points_path, allow_pickle=True).item()
        points = points_data["points"]  # Shape: [1, num_points, 2]
        labels = points_data["labels"]  # Shape: [1, num_points]

        image = torch.tensor(image.transpose(2, 0, 1) / 255.0, dtype=torch.float32)  # [C, H, W]
        mask = torch.tensor(mask / 255.0, dtype=torch.float32)  # [H, W]
        points = torch.tensor(points, dtype=torch.float32)  # [1, num_points, 2]
        labels = torch.tensor(labels, dtype=torch.float32)  # [1, num_points]

        return image, mask, points, labels


# 测试代码
# if __name__ == "__main__":
#     # 数据路径
#     image_dir = r"E:\DP\1sam_point_updata\data\output\images"
#     mask_dir = r"E:\DP\1sam_point_updata\data\output\masks"
#     points_dir = r"E:\DP\1sam_point_updata\data\output\points"

#     # 创建数据集实例
#     dataset = SAMCompatibleDataset(image_dir, mask_dir, points_dir)

#     # 验证数据集的长度
#     print(f"Dataset size: {len(dataset)}")

#     # 加载一个样本并验证
#     image, mask, points, labels = dataset[0]

#     # 打印样本的基本信息
#     print(f"Image shape: {image.shape} (C, H, W)")
#     print(f"Mask shape: {mask.shape} (H, W)")
#     print(f"Points shape: {points.shape} (batch, num_points, 2)")
#     print(f"Labels shape: {labels.shape} (num_points)")
