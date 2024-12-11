"""
Author: YiJin Li
Email: your.email@example.com
Date: 2024-11-28
"""

import os
import torch
import torch.nn.functional as F
from model import SAMFineTuner
from tqdm import tqdm
import tifffile as tiff
import numpy as np


def pad_to_1024(image, mask=None):
    """
    同步填充图像和掩码到 1024×1024
    :param image: 图像，支持输入形状 (H, W, C)、(H, W) 或 (C, H, W)
    :param mask: 掩码，支持输入形状 (H, W)
    :return: 填充后的图像和掩码
    """
    print(f"Original image shape: {image.shape}")  # 打印输入形状

    # 处理图像为 (C, H, W)
    if image.ndim == 4:  # 如果是 (1, H, W, C)
        image = image.squeeze(0)  # 移除 batch 维度，变为 (H, W, C)
    if image.ndim == 3 and image.shape[-1] in [1, 3]:  # 如果是 (H, W, C)
        image = np.transpose(image, (2, 0, 1))  # 转为 (C, H, W)
    elif image.ndim == 2:  # 如果是 (H, W)
        image = image[np.newaxis, ...]  # 添加通道维度，变为 (1, H, W)

    print(f"Processed image shape (C, H, W): {image.shape}")

    # 确保图像现在是 (C, H, W)
    _, h, w = image.shape
    pad_h = (1024 - h) // 2
    pad_w = (1024 - w) // 2
    padded_image = F.pad(torch.tensor(image), (pad_w, pad_w, pad_h, pad_h), mode="constant", value=0)

    if mask is not None:
        # 处理掩码为 (H, W)
        print(f"Original mask shape: {mask.shape}")  # 打印掩码输入形状
        if mask.ndim == 3:  # 如果是 (C, H, W)
            mask = mask.squeeze(0)  # 移除通道维度，变为 (H, W)
        padded_mask = F.pad(torch.tensor(mask), (pad_w, pad_w, pad_h, pad_h), mode="constant", value=0)
        return padded_image, padded_mask

    return padded_image, None


def compute_metrics(predicted_masks, true_masks, threshold=0.5):
    """
    计算评价指标：IoU 和 Dice 系数
    """
    print(f"predicted_masks shape: {predicted_masks.shape}")
    print(f"true_masks shape: {true_masks.shape}")

    # 确保 `predicted_masks` 是 3D 或 4D
    if predicted_masks.ndim == 4:  # (B, 1, H, W)
        predicted_masks = predicted_masks.squeeze(1)  # (B, H, W)
    elif predicted_masks.ndim == 3:  # (H, W)
        predicted_masks = predicted_masks.unsqueeze(0)  # (1, H, W)

    # 确保 `true_masks` 是 3D
    if true_masks.ndim == 2:  # (H, W)
        true_masks = true_masks.unsqueeze(0)  # (1, H, W)

    # 检查形状是否匹配
    if predicted_masks.shape != true_masks.shape:
        raise ValueError(
            f"Shape mismatch between predicted_masks {predicted_masks.shape} and true_masks {true_masks.shape}"
        )

    # 二值化预测结果
    predicted_masks = (torch.sigmoid(predicted_masks) > threshold).float()

    # 计算交并比和 Dice 系数
    intersection = (predicted_masks * true_masks).sum(dim=(1, 2))
    union = predicted_masks.sum(dim=(1, 2)) + true_masks.sum(dim=(1, 2)) - intersection
    iou = (intersection / union).mean().item()

    dice = (2 * intersection / (predicted_masks.sum(dim=(1, 2)) + true_masks.sum(dim=(1, 2)))).mean().item()
    return iou, dice


def predict(model, image_paths, points_paths, mask_paths=None, device="cuda"):
    """
    批量预测
    """
    model.eval()
    predicted_masks_list = []
    ious, dices = [], []

    with torch.no_grad():
        for image_path, points_path, mask_path in tqdm(
            zip(image_paths, points_paths, mask_paths if mask_paths else [None] * len(image_paths)),
            desc="Predicting",
            total=len(image_paths),
        ):
            # 加载图像
            image = tiff.imread(image_path)
            print(f"Loaded image shape: {image.shape}")

            # 转换图像类型为 float32 并归一化到 [0, 1]
            image = image.astype(np.float32) / 255.0

            # 加载点提示
            points_data = np.load(points_path, allow_pickle=True).item()
            points = points_data["points"]
            labels = points_data["labels"]
            points_tensor = torch.tensor(points, dtype=torch.float32).unsqueeze(0)  # Shape: (1, N, 2)
            labels_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(0)  # Shape: (1, N)

            print(f"Original points_tensor shape: {points_tensor.shape}")
            print(f"Original labels_tensor shape: {labels_tensor.shape}")

            # 加载掩码（如果提供）
            mask = None
            if mask_path:
                mask = tiff.imread(mask_path)
                print(f"Loaded mask shape: {mask.shape}")

            # 将数据移动到设备
            image, mask = pad_to_1024(image, mask)
            image = image.unsqueeze(0)  # 添加 batch 维度，确保 (B, C, H, W)
            image = image.to(device)
            points_tensor = points_tensor.to(device)
            labels_tensor = labels_tensor.to(device)
            if mask is not None:
                mask = mask.to(device)

            # 调整 points_tensor 和 labels_tensor 形状以符合模型要求
            if points_tensor.ndim == 5:
                points_tensor = points_tensor.squeeze(2)  # Shape: (1, 1, N, 2)
            if labels_tensor.ndim == 4:
                labels_tensor = labels_tensor.squeeze(2)  # Shape: (1, 1, N)

            print(f"Processed points_tensor shape after squeeze: {points_tensor.shape}")
            print(f"Processed labels_tensor shape after squeeze: {labels_tensor.shape}")

            # 预测
            predicted_masks = model(image, points_tensor, labels_tensor)

            # 如果提供了真实掩码，计算指标
            if mask is not None:
                iou, dice = compute_metrics(predicted_masks, mask)
                ious.append(iou)
                dices.append(dice)

            # 存储预测结果
            predicted_masks_list.append(predicted_masks.squeeze(0).cpu().numpy())

    return predicted_masks_list, ious, dices


def save_predictions(predicted_masks_list, output_dir, image_paths):
    """
    保存预测掩码为单波段 .tif 文件，确保与原始图像尺寸一致
    """
    os.makedirs(output_dir, exist_ok=True)
    for i, (predicted_mask, image_path) in enumerate(zip(predicted_masks_list, image_paths)):
        # 获取原始图像的尺寸（H, W）
        image = tiff.imread(image_path)
        h, w = image.shape[:2]  # 获取图像的高和宽

        # 去除填充部分，恢复到原始尺寸
        predicted_mask_cropped = predicted_mask[:h, :w]  # 只保留与原图像相同的区域

        output_path = os.path.join(output_dir, f"predicted_mask_{i + 1}.tif")
        tiff.imsave(output_path, predicted_mask_cropped.astype(np.float32))  # 保存为.tif格式

        print(f"Saved {output_path} with shape {predicted_mask_cropped.shape}")


if __name__ == "__main__":
    # 输入文件夹路径
    image_dir = r"E:\DP\1sam_point_updata\val\image"  # 输入图像文件夹路径
    points_dir = r"E:\DP\1sam_point_updata\val\point"  # 点提示文件夹路径
    mask_dir = r"E:\DP\1sam_point_updata\val\mask"  # 输入掩码文件夹路径（可选）
    output_dir = r"E:\DP\1sam_point_updata\final1"  # 预测结果输出文件夹路径

    # 获取所有文件路径
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".tif")]
    points_paths = [os.path.join(points_dir, f.replace(".tif", ".npy")) for f in os.listdir(image_dir) if f.endswith(".tif")]
    mask_paths = [os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith(".tif")] if os.path.exists(mask_dir) else None

    # 模型初始化
    model = SAMFineTuner()  # 这里假设您已经定义了 SAMFineTuner 类
    model.load_state_dict(torch.load(r"E:\DP\1sam_point_updata\3\sam_epoch_50.pth"))
    model = model.cuda()  # 使用 GPU

    # 预测
    predicted_masks_list, ious, dices = predict(model, image_paths, points_paths, mask_paths)

    # 保存预测结果
    save_predictions(predicted_masks_list, output_dir, image_paths)

    # 打印评估指标
    if ious and dices:
        print(f"Mean IoU: {np.mean(ious)}")
        print(f"Mean Dice: {np.mean(dices)}")
