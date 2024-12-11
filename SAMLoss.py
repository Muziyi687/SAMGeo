"""
Author: YiJin Li
Email: your.email@example.com
Date: 2024-11-28
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SAMLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super(SAMLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.bce_loss = nn.BCEWithLogitsLoss()

    def dice_loss(self, predicted_mask, mask):
        smooth = 1e-5
        predicted_mask = torch.sigmoid(predicted_mask)
        intersection = (predicted_mask * mask).sum(dim=(1, 2, 3))
        union = predicted_mask.sum(dim=(1, 2, 3)) + mask.sum(dim=(1, 2, 3))
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()

    def forward(self, predicted_masks, masks, points=None, labels=None):
        # 确保 predicted_masks 是 4D 张量
        if predicted_masks.dim() == 3:  # 如果是 3D，则添加 batch 维度
            predicted_masks = predicted_masks.unsqueeze(0)

        if masks.dim() == 3:  # 如果是 3D，则添加 channel 维度
            masks = masks.unsqueeze(1)

        if predicted_masks.size(1) > 1:  # 如果有多个通道
            predicted_masks = predicted_masks[:, 0:1, :, :]  # 取第一个通道

        predicted_masks = F.interpolate(predicted_masks, size=masks.shape[2:], mode="bilinear", align_corners=False)

        bce = self.bce_loss(predicted_masks, masks)
        dice = self.dice_loss(predicted_masks, masks)

        return self.alpha * bce + self.beta * dice
