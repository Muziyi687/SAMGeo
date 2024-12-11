import cv2
import numpy as np
import matplotlib.pyplot as plt


def visualize_data(image_path, mask_path, points_path):
    """
    可视化数据：显示图像、掩码和点提示
    :param image_path: 图像路径
    :param mask_path: 掩码路径
    :param points_path: 点提示路径 (包含 points 和 labels)
    """
    # 读取图像和掩码
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    # 加载点提示数据
    points_data = np.load(points_path, allow_pickle=True).item()
    points, labels = points_data['points'], points_data['labels']

    # 输出 points 和 labels 的维度
    print(f"Points shape: {points.shape}")  # 应为 [1, num_points, 2]
    print(f"Labels shape: {labels.shape}")  # 应为 [1, num_points]

    # 检查并调整点提示和标签的维度
    if points.ndim == 3 and points.shape[0] == 1:  # Shape: [1, num_points, 2]
        points = points[0]  # 转为 [num_points, 2]
    if labels.ndim == 2 and labels.shape[0] == 1:  # Shape: [1, num_points]
        labels = labels[0]  # 转为 [num_points]

    # 确保点提示和标签的点数匹配
    assert points.shape[0] == labels.shape[0], "点提示和标签的数量不一致！"

    # 可视化图像和点提示
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Image with Points")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    for (y, x), label in zip(points, labels):
        color = 'g' if label == 1 else 'r'  # 前景点用绿色，背景点用红色
        plt.scatter(x, y, c=color, s=10)

    # 可视化掩码
    plt.subplot(1, 2, 2)
    plt.title("Mask")
    plt.imshow(mask, cmap='gray')
    plt.show()


# 示例：验证编号为1的数据
visualize_data(
    r"E:\DP\1sam_point_updata\data\output\images\2.tif",
    r"E:\DP\1sam_point_updata\data\output\masks\2.tif",
    r"E:\DP\1sam_point_updata\data\output\points\2.npy"
)
