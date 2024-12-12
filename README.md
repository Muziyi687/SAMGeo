# SAMGeo
fine-tuning sam to plot extraction
# SAM Project Code Documentation

## 概述

本项目基于 Segment Anything Model (SAM) 对遥感影像进行定制化分割训练与推理，并配套了从数据预处理到模型微调、预测的完整流程。主要功能包括：  
- **数据集预处理**：可对大幅遥感影像进行裁剪与增强，并自动生成相应的点提示数据。  
- **模型微调**：在预训练的 SAM 模型基础上，通过冻结部分参数，定制化微调用于特定分割任务。  
- **自定义损失函数**：将 BCE 和 Dice Loss 相结合，从而提升分割精度和稳定性。  
- **预测模块**：基于微调后的模型对新图像进行分割预测，并计算相关性能指标。

以下将详细介绍各个模块的实现与使用方法。

---

## 文件结构

- **SAMDataset.py**：定义兼容 SAM 的自定义数据集类，便于数据加载和处理。  
- **SAMLoss.py**：实现自定义损失函数（BCE 与 Dice Loss 的组合）。  
- **model.py**：提供基于预训练 SAM 模型的微调机制。  
- **train.py**：训练脚本，包括训练、验证流程以及指标计算和模型保存。  
- **predict.py**：预测脚本，可对批量图像进行分割预测并保存结果。  
- **PreProcessData.py**：数据预处理模块，用于裁剪大图、数据增强和生成点提示数据。

---

## 模块详解

### 1. 数据集加载模块
**文件：SAMDataset.py**

**核心类：** `SAMCompatibleDataset`  
- 负责加载图像、掩码及点提示数据。  
- 自动调节数据尺寸以适配 SAM 模型输入要求。

**关键方法：**  
- `__init__`: 初始化数据集，检查数据文件数量并设定基本参数。  
- `__getitem__`: 根据索引返回图像、掩码、点提示和对应的标签数据。

**输入输出：**  
- **输入**：图像文件 (`image_dir`)，掩码文件 (`mask_dir`)，点提示文件 (`points_dir`)。  
- **输出**：`torch.Tensor` 格式的图像，掩码数据，点提示以及标签信息。

---

### 2. 自定义损失函数
**文件：SAMLoss.py**

**核心类：** `SAMLoss`  
- 同时使用 `BCEWithLogitsLoss` 和自定义的 `dice_loss`。  
- 支持通过 `alpha` 和 `beta` 参数调整两种损失的权重比例。

**关键方法：**  
- `dice_loss`: 计算 Dice Loss，用于衡量预测分割和真实掩码的相似度。  
- `forward`: 将 BCE 和 Dice Loss 加权合成为最终损失。

**应用场景：**  
- 在 `train.py` 中对训练集和验证集的预测结果计算损失，从而指导模型参数更新。

---

### 3. 微调模型模块
**文件：model.py**

**核心类：** `SAMFineTuner`  
- 基于 `transformers` 库中的 SAM 模型进行定制化微调。  
- 支持冻结 `vision_encoder` 等部分参数，仅对特定模块进行更新。

**关键方法：**  
- `__init__`: 初始化预训练模型并设置冻结策略。  
- `forward`: 完成对输入图像和点提示的前向传播，输出分割结果。

**输出：**  
- 与输入图像尺寸一致的预测分割掩码。

---

### 4. 训练模块
**文件：train.py**

**功能概述：**  
实现训练和验证的全流程，包括数据加载、模型训练、验证指标计算和模型保存。

**核心函数：**  
- `pad_to_1024`: 将图像和掩码填充或缩放到 1024×1024 的统一尺寸。  
- `compute_metrics`: 计算 IoU 和 Dice 等分割性能指标。  
- `collate_fn`: 自定义的批处理函数，用于数据加载器。

**训练流程：**  
1. 数据加载：使用 `SAMCompatibleDataset` 获取训练和验证数据。  
2. 模型构建：利用 `SAMFineTuner` 初始化微调模型，配合 `SAMLoss` 和优化器。  
3. 训练迭代：前向传播 -> 计算损失 -> 反向传播 -> 更新参数。  
4. 验证阶段：计算验证集损失和分割指标（IoU、Dice）。  
5. 模型保存：每个 Epoch 完成后保存当前模型权重。  
6. 可视化：绘制训练与验证损失曲线，帮助监控训练过程。

**超参数：**  
- `num_epochs`: 训练轮数。  
- `batch_size`: 批量大小。  
- `base_learning_rate`: 基础学习率。  
- `mask_decoder_lr`: 掩码解码器的学习率。

---

### 5. 数据预处理模块
**文件：PreProcessData.py**

**核心类：** `DataPreprocessorTIF`  
- 针对遥感影像的预处理模块，包括图像与掩码的裁剪、数据增强和点提示生成。

**关键方法：**  
- `crop_image_and_mask`: 将大幅遥感图像及其对应掩码裁剪为小块。  
- `augment_image_and_mask`: 对图像和掩码进行随机翻转、旋转等增强操作。  
- `generate_point_prompt`: 基于掩码生成正负样本点提示数据。  
- `process_and_save`: 批量处理数据并将结果保存到指定目录下的 `images`、`masks` 和 `points` 文件夹。

---

### 6. 预测模块
**文件：predict.py**

**核心功能：**  
- 对输入图像进行填充并通过微调模型进行分割预测。  
- 支持批量预测与指标计算（IoU、Dice）。  
- 保存预测结果为单波段 `.tif` 文件。

**关键函数：**  
- `pad_to_1024`: 对输入图像和掩码进行填充。  
- `compute_metrics`: 计算预测结果与真实标签之间的 IoU 和 Dice 分数。  
- `predict`: 批量处理输入图像，返回预测掩码和性能指标。  
- `save_predictions`: 将预测掩码保存到指定目录。

---

### 7. 二值化
**文件：afterpreprocess.py**

**核心功能**
- 通过阈值设定对图像进行二分类。

**使用方法：**  
1. 加载已训练好的模型权重。  
2. 输入测试图像和点提示路径。  
3. 调用 `predict` 和 `save_predictions` 获取并保存预测结果。

---

**未完成工作：**
我们没有生成点的模型，所以我们是通过mask来生成点的，如果您有生成点的模型，那代码会更完善。

## 总结

此代码库为遥感影像的分割任务提供了完整的处理流程：从数据预处理、模型微调再到结果预测和评估。各个模块之间具备较高的独立性和灵活性，便于根据实际需求进行扩展和优化。


希望本说明文档对您理解和使用本项目有所帮助！
