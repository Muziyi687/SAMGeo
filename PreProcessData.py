"""
Author: YiJin Li
Email: your.email@example.com
Date: 2024-11-28
"""

import os
import rasterio
import numpy as np
import random
import cv2

class DataPreprocessorTIF:
    def __init__(self, image_dir, mask_dir, output_dir, crop_size=1024, overlap=0.5, num_points=10):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.output_dir = output_dir
        self.crop_size = crop_size
        self.stride = int(crop_size * (1 - overlap))
        self.num_points = num_points

    def load_tif(self, filepath):
        with rasterio.open(filepath) as src:
            return src.read([1, 2, 3]).transpose(1, 2, 0)

    def load_mask(self, filepath):
        with rasterio.open(filepath) as src:
            return src.read(1)

    def crop_image_and_mask(self, image, mask):
        h, w = image.shape[:2]
        cropped_images = []
        cropped_masks = []
        for i in range(0, h - self.crop_size + 1, self.stride):
            for j in range(0, w - self.crop_size + 1, self.stride):
                cropped_image = image[i:i+self.crop_size, j:j+self.crop_size]
                cropped_mask = mask[i:i+self.crop_size, j:j+self.crop_size]
                if cropped_image.shape[0] == self.crop_size and cropped_image.shape[1] == self.crop_size:
                    cropped_images.append(cropped_image)
                    cropped_masks.append(cropped_mask)
        return cropped_images, cropped_masks

    def augment_image_and_mask(self, image, mask):
        if random.random() > 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
        if random.random() > 0.5:
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)
        if random.random() > 0.5:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            mask = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
        return image, mask

    def generate_point_prompt(self, mask):
        foreground_indices = np.argwhere(mask == 255)
        background_indices = np.argwhere(mask == 0)

        if len(foreground_indices) < self.num_points:
            print(f"Warning: 前景点不足 {self.num_points} 个，将只生成 {len(foreground_indices)} 个前景点")
        if len(background_indices) < self.num_points:
            print(f"Warning: 背景点不足 {self.num_points} 个，将只生成 {len(background_indices)} 个背景点")

        fg_points = foreground_indices[np.random.choice(
            foreground_indices.shape[0],
            min(self.num_points, len(foreground_indices)),
            replace=False
        )]
        bg_points = background_indices[np.random.choice(
            background_indices.shape[0],
            min(self.num_points, len(background_indices)),
            replace=False
        )]

        points = np.vstack((fg_points, bg_points))  # [num_points, 2]
        labels = np.array([1] * len(fg_points) + [0] * len(bg_points))  # 前景点=1，背景点=0

        # 调整维度
        points = points[np.newaxis, ...]  # [1, num_points, 2]
        labels = labels[np.newaxis, ...]  # [1, num_points]

        return points, labels

    def process_and_save(self):
        image_files = [f for f in os.listdir(self.image_dir) if f.endswith('.tif')]
        mask_files = [f for f in os.listdir(self.mask_dir) if f.endswith('.tif')]
        image_files.sort()
        mask_files.sort()

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "masks"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "points"), exist_ok=True)

        counter = 1

        for img_file, msk_file in zip(image_files, mask_files):
            image = self.load_tif(os.path.join(self.image_dir, img_file))
            mask = self.load_mask(os.path.join(self.mask_dir, msk_file))

            cropped_images, cropped_masks = self.crop_image_and_mask(image, mask)

            for cropped_image, cropped_mask in zip(cropped_images, cropped_masks):
                img_path = os.path.join(self.output_dir, "images", f"{counter}.tif")
                mask_path = os.path.join(self.output_dir, "masks", f"{counter}.tif")
                points_path = os.path.join(self.output_dir, "points", f"{counter}.npy")

                with rasterio.open(
                    img_path,
                    "w",
                    driver="GTiff",
                    height=cropped_image.shape[0],
                    width=cropped_image.shape[1],
                    count=3,
                    dtype=cropped_image.dtype
                ) as dst:
                    dst.write(cropped_image.transpose(2, 0, 1))

                with rasterio.open(
                    mask_path,
                    "w",
                    driver="GTiff",
                    height=cropped_mask.shape[0],
                    width=cropped_mask.shape[1],
                    count=1,
                    dtype=cropped_mask.dtype
                ) as dst:
                    dst.write(cropped_mask, 1)

                points, labels = self.generate_point_prompt(cropped_mask)
                np.save(points_path, {"points": points, "labels": labels})

                aug_image, aug_mask = self.augment_image_and_mask(cropped_image, cropped_mask)

                counter += 1
                img_path = os.path.join(self.output_dir, "images", f"{counter}.tif")
                mask_path = os.path.join(self.output_dir, "masks", f"{counter}.tif")
                points_path = os.path.join(self.output_dir, "points", f"{counter}.npy")

                with rasterio.open(
                    img_path,
                    "w",
                    driver="GTiff",
                    height=aug_image.shape[0],
                    width=aug_image.shape[1],
                    count=3,
                    dtype=aug_image.dtype
                ) as dst:
                    dst.write(aug_image.transpose(2, 0, 1))

                with rasterio.open(
                    mask_path,
                    "w",
                    driver="GTiff",
                    height=aug_mask.shape[0],
                    width=aug_mask.shape[1],
                    count=1,
                    dtype=aug_mask.dtype
                ) as dst:
                    dst.write(aug_mask, 1)

                points, labels = self.generate_point_prompt(aug_mask)
                np.save(points_path, {"points": points, "labels": labels})

                counter += 1

        print(f"数据预处理完成，已保存 {counter - 1} 个裁剪块。")


# 使用示例
image_dir = r"E:\DP\1sam_point_updata\data\image"
mask_dir = r"E:\DP\1sam_point_updata\data\mask"
output_dir = r"E:\DP\1sam_point_updata\data\output"

preprocessor = DataPreprocessorTIF(image_dir, mask_dir, output_dir, crop_size=1024, overlap=0.5, num_points=10)
preprocessor.process_and_save()
