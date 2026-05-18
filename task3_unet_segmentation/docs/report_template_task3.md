# 任务 3 报告草稿模板：从零搭建 U-Net 与损失函数工程

> 请把“待填写”替换为真实信息。不要编造结果，所有数值以 `results/summary.csv` 和各实验 `metrics_best.json` 为准。

## 1. 任务简介

本任务从零搭建一个经典 U-Net 语义分割模型，不使用任何预训练权重，在 Stanford Background Dataset 上进行像素级训练。为了分析损失函数对类别不平衡和收敛的影响，分别使用 Cross-Entropy Loss、手动实现的 Dice Loss、Cross-Entropy + Dice Loss 三种配置训练模型，并比较验证集 mIoU。

## 2. 数据集

- 数据集：Stanford Background Dataset
- 类别数：8
- 类别名称：sky、tree、road、grass、water、building、mountain、foreground
- train 数量：待填写
- val 数量：待填写
- test 数量：待填写
- 输入尺寸：256 × 256
- 数据增强：随机水平翻转、亮度和对比度轻微扰动

## 3. U-Net 网络结构

本实验手写实现 U-Net，主要由以下部分组成：

1. 编码器：使用 `DoubleConv + MaxPool` 逐层下采样，通道数逐层增加。
2. Bottleneck：最深层卷积块提取高层语义特征。
3. 解码器：使用转置卷积上采样，并与编码器对应层特征进行拼接。
4. Skip Connection：将浅层空间细节传递到解码端，帮助恢复边界和小目标。
5. 输出层：1×1 卷积将特征映射到 8 类 logits。

本实验不加载任何预训练权重，所有参数均随机初始化。

## 4. 损失函数

### 4.1 Cross-Entropy Loss

标准多分类像素级交叉熵，用于监督每个像素的类别预测。

### 4.2 Dice Loss

Dice Loss 由代码手动实现。对 logits 做 softmax 得到每类概率，将标签转换为 one-hot，然后计算每个类别的 Dice 系数并取平均：

```text
Dice = (2 * intersection + smooth) / (prediction + target + smooth)
Dice Loss = 1 - mean(Dice)
```

它更关注预测区域与真实区域的重叠，对前景/背景像素不平衡更敏感。

### 4.3 CE + Dice

组合损失为：

```text
Loss = CrossEntropyLoss + DiceLoss
```

期望同时保留 CE 的像素级分类稳定性和 Dice 对区域重叠的优化效果。

## 5. 实验设置

| 实验名 | loss | epoch | batch size | image size | base channels | lr | optimizer | metric |
|---|---|---:|---:|---:|---:|---:|---|---|
| 01_unet_ce | CE | 80 | 8 | 256 | 32 | 3e-4 | AdamW | mIoU |
| 02_unet_dice | Dice | 80 | 8 | 256 | 32 | 3e-4 | AdamW | mIoU |
| 03_unet_ce_dice | CE+Dice | 80 | 8 | 256 | 32 | 3e-4 | AdamW | mIoU |

## 6. 实验结果

> 从 `results/summary.md` 或 `results/summary.csv` 复制结果表。

| 实验名 | loss | best epoch | best val loss | best val mIoU | best val pixel acc | 权重路径 |
|---|---|---:|---:|---:|---:|---|
| 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |

## 7. 曲线与预测可视化

插入以下图：

- `results/train_loss_curves.png`
- `results/val_loss_curves.png`
- `results/val_miou_curves.png`
- `results/val_pixel_acc_curves.png`
- `results/summary_bar.png`
- `results/predictions/prediction_*.jpg`

## 8. 分析

### 8.1 CE Loss 表现

待填写：分析 CE 的收敛速度、验证集 mIoU、像素精度，以及它是否更偏向大面积类别。

### 8.2 Dice Loss 表现

待填写：分析 Dice 是否改善小类别或前景区域，是否训练更不稳定。

### 8.3 CE + Dice 表现

待填写：分析组合损失是否取得最高 mIoU，原因是 CE 提供稳定的逐像素分类监督，Dice 强化区域重叠。

### 8.4 最佳模型

待填写：根据验证集 mIoU 选择最佳实验和权重。

## 9. 结论

待填写：总结三种 loss 的差异，说明最终推荐哪种配置。

## 10. 提交信息

- GitHub repo 链接：待填写
- 模型权重网盘链接：待填写
- 小组成员与分工：待填写
