"""
Author: YiJin Li
Email: your.email@example.com
Date: 2024-11-28
"""

import torch
from transformers import SamModel, SamProcessor
import torch.nn as nn
import torch.nn.functional as F


class SAMFineTuner(nn.Module):
    def __init__(self, model_name="facebook/sam-vit-base"):
        """
        初始化 SAM 微调模型
        :param model_name: SAM 模型名称，默认为 ViT-B 版本
        """
        super(SAMFineTuner, self).__init__()
        # 加载预训练的 SAM 模型
        self.model = SamModel.from_pretrained(model_name)
        self.processor = SamProcessor.from_pretrained(model_name)

        for name, param in self.model.named_parameters():
            if "vision_encoder" in name:  
                param.requires_grad = False  # 冻结
            else:
                param.requires_grad = True  # 解冻其他部分
                #print(f"Unfrozen parameter: {name}")

    def forward(self, images, points=None, labels=None):
        """
        前向传播
        :param images: 输入图像 (B, C, H, W)
        :param points: 点提示坐标 (B, 1, N, 2)，可选
        :param labels: 点提示标签 (B, 1, N)，可选
        :return: 模型输出的分割掩码 (B, 1, H, W)
        """
        # 处理输入
        inputs = {"pixel_values": images}
        if points is not None:
            inputs["input_points"] = points
        if labels is not None:
            inputs["input_labels"] = labels

        # 前向传播
        outputs = self.model(**inputs)

        # 提取模型输出的分割掩码
        predicted_masks = outputs.pred_masks  # 原始输出

        # 检查维度并提取正确的掩码数据
        if predicted_masks.dim() == 5 and predicted_masks.shape[2] == 3:  # [B, 1, 3, H, W]
            predicted_masks = predicted_masks[:, :, 0, :, :]  # 提取前景掩码 -> [B, 1, H, W]
        elif predicted_masks.dim() != 4:  # 确保掩码维度为 [B, 1, H, W]
            raise ValueError(f"Unexpected shape for predicted_masks: {predicted_masks.shape}")

        # 调整到输入图像的尺寸
        predicted_masks = F.interpolate(
            predicted_masks,
            size=(images.shape[2], images.shape[3]),  # (H, W)
            mode="bilinear",
            align_corners=False,
        )

        return predicted_masks


# if __name__ == "__main__":
#     # 检查 CUDA 是否可用
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"Using device: {device}")

#     # 创建模型实例
#     model_name = "facebook/sam-vit-base"
#     model = SAMFineTuner(model_name=model_name).to(device)

#     # 模拟输入进行测试
#     images = torch.rand(2, 3, 1024, 1024).to(device)  # 模拟输入图像 (B, C, H, W)
#     points = torch.rand(2, 1, 10, 2).to(device)       # 模拟点提示 (B, 1, N, 2)
#     labels = torch.randint(0, 2, (2, 1, 10)).to(device)  # 模拟点提示标签 (B, 1, N)

#     # 测试模型
#     predicted_masks = model(images, points, labels)
#     print(f"Predicted masks shape: {predicted_masks.shape}")  # 应为 (B, 1, 1024, 1024)
