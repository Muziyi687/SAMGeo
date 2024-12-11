"""
Author: YiJin Li
Email: your.email@example.com
Date: 2024-11-28
"""

import os
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt  # 用于绘制损失曲线

# Import necessary modules
from SAMDataset import SAMCompatibleDataset
from SAMLoss import SAMLoss
from model import SAMFineTuner
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 禁止 TensorFlow 显示 INFO 和 WARNING 日志
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # 禁用 oneDNN 自定义操作


def pad_to_1024(image, mask):
    """
    同步填充图像和掩码到1024×1024，仅当图像或掩码小于1024时才填充
    :param image: 图像 (C, H, W)
    :param mask: 掩码 (H, W)
    :return: 填充后的图像和掩码
    """
    _, h, w = image.shape
    if h < 1024 or w < 1024:
        pad_h = (1024 - h) // 2
        pad_w = (1024 - w) // 2
        padded_image = F.pad(image, (pad_w, pad_w, pad_h, pad_h), mode="constant", value=0)
        padded_mask = F.pad(mask, (pad_w, pad_w, pad_h, pad_h), mode="constant", value=0)
        return padded_image, padded_mask
    else:
        return image, mask


def compute_metrics(predicted_masks, true_masks, threshold=0.5):
    """
    计算评价指标：IoU 和 Dice 系数
    """
    if predicted_masks.ndim == 4 and predicted_masks.size(1) == 1:
        predicted_masks = predicted_masks.squeeze(1)
    if true_masks.ndim == 4 and true_masks.size(1) == 1:
        true_masks = true_masks.squeeze(1)

    predicted_masks = (torch.sigmoid(predicted_masks) > threshold).float()
    intersection = (predicted_masks * true_masks).sum(dim=(1, 2))
    union = predicted_masks.sum(dim=(1, 2)) + true_masks.sum(dim=(1, 2)) - intersection
    iou = (intersection / union).mean().item()

    dice = (2 * intersection / (predicted_masks.sum(dim=(1, 2)) + true_masks.sum(dim=(1, 2)))).mean().item()
    return iou, dice


def collate_fn(batch):
    """
    自定义 collate_fn 处理 DataLoader 中的批次数据，确保所有数据都填充到一致的大小
    """
    images, masks, points, labels = zip(*batch)
    
    # 对每个样本进行填充，确保它们的大小为 1024x1024
    padded_images = []
    padded_masks = []
    for image, mask in zip(images, masks):
        padded_image, padded_mask = pad_to_1024(image, mask)
        padded_images.append(padded_image)
        padded_masks.append(padded_mask)
    
    # 将处理后的图像和掩码堆叠成一个 batch
    images = torch.stack(padded_images)
    masks = torch.stack(padded_masks)
    
    # 将 points 和 labels 也堆叠成一个 batch
    points = torch.stack(points)
    labels = torch.stack(labels)
    
    return images, masks, points, labels


if __name__ == "__main__":
    # Paths
    image_dir = r"E:\DP\1sam_point_updata\data\output\images"
    mask_dir = r"E:\DP\1sam_point_updata\data\output\masks"
    points_dir = r"E:\DP\1sam_point_updata\data\output\points"
    checkpoint_dir = "./1"

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Parameters
    num_epochs = 3
    batch_size = 8
    base_learning_rate = 1e-4
    mask_decoder_lr = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset and DataLoader
    dataset = SAMCompatibleDataset(image_dir, mask_dir, points_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)

    # Model
    model = SAMFineTuner(model_name="facebook/sam-vit-base").to(device)
    loss_fn = SAMLoss(alpha=0.5, beta=0.5)

    # Optimizer with parameter groups
    optimizer = Adam([{
        "params": [p for n, p in model.named_parameters() if "mask_decoder" in n],
        "lr": mask_decoder_lr,
    }, {
        "params": [p for n, p in model.named_parameters() if "mask_decoder" not in n and p.requires_grad],
        "lr": base_learning_rate,
    }])

    # Scheduler for learning rate adjustment
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    # Log parameter info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}, Trainable parameters: {trainable_params}")

    # Lists to store loss values for plotting
    train_losses = []
    val_losses = []

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for images, masks, points, labels in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            images, masks, points, labels = images.to(device), masks.to(device), points.to(device), labels.to(device)

            # Forward and loss computation
            predicted_masks = model(images, points, labels)
            loss = loss_fn(predicted_masks, masks, points, labels)
            epoch_loss += loss.item()

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Learning rate adjustment
        scheduler.step()

        # Log training loss
        avg_train_loss = epoch_loss / len(dataloader)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            total_iou, total_dice = 0.0, 0.0
            for images, masks, points, labels in dataloader:
                images, masks, points, labels = images.to(device), masks.to(device), points.to(device), labels.to(device)

                predicted_masks = model(images, points, labels)
                val_loss += loss_fn(predicted_masks, masks, points, labels).item()

                iou, dice = compute_metrics(predicted_masks, masks)
                total_iou += iou
                total_dice += dice

            avg_val_loss = val_loss / len(dataloader)
            avg_iou = total_iou / len(dataloader)
            avg_dice = total_dice / len(dataloader)

            val_losses.append(avg_val_loss)

            print(f"Validation Loss: {avg_val_loss:.4f}, IoU: {avg_iou:.4f}, Dice: {avg_dice:.4f}")

        # Save model checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"sam_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    # Plot training and validation loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', color='blue')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()
