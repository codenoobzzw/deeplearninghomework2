# 任务 3：从零搭建 U-Net 与损失函数工程

## 1. 本任务需要完成什么

本任务对应作业“从零搭建与损失函数工程：图像分割模型的像素级训练”。你需要：

1. 不使用任何预训练权重。
2. 使用 PyTorch 基础 API 手写经典 U-Net。
3. 网络必须包含下采样编码器、上采样解码器和 skip connection。
4. 使用 Stanford Background Dataset。
5. 手动实现 Dice Loss。
6. 分别训练三种 loss 配置：CE、Dice、CE+Dice。
7. 对比验证集 mIoU。

## 2. 目录结构

```text
task3_unet_segmentation/
├── configs/task3_unet_plan.yaml
├── scripts/
│   ├── check_env.py
│   ├── prepare_stanford_background.py
│   ├── data.py
│   ├── model.py
│   ├── losses.py
│   ├── metrics.py
│   ├── train.py
│   ├── run_experiments.py
│   ├── evaluate.py
│   ├── visualize_predictions.py
│   └── summarize_results.py
├── docs/report_template_task3.md
└── README_task3.md
```

## 3. 数据准备

Kaggle 自动下载：

```bash
python scripts/prepare_stanford_background.py --kaggle --out-dir data/stanford_background
```

使用本地上传数据：

```bash
python scripts/prepare_stanford_background.py \
  --source-dir /path/to/stanford-background-dataset \
  --out-dir data/stanford_background
```

处理后目录：

```text
data/stanford_background/
├── processed/images/*.jpg
├── processed/masks/*.png
├── splits/train.txt
├── splits/val.txt
├── splits/test.txt
└── dataset_info.json
```

## 4. 冒烟测试

```bash
python scripts/train.py \
  --data-dir data/stanford_background \
  --loss ce \
  --epochs 1 \
  --batch-size 2 \
  --image-size 256 \
  --base-channels 16 \
  --output-dir runs/smoke_ce \
  --amp \
  --tracker none
```

## 5. 正式实验

```bash
python scripts/run_experiments.py \
  --plan configs/task3_unet_plan.yaml \
  --data-dir data/stanford_background \
  --runs-dir runs \
  --tracker wandb \
  --wandb-mode offline \
  --amp \
  --skip-existing \
  --continue-on-error
```

## 6. 汇总和可视化

```bash
python scripts/summarize_results.py --runs-dir runs --out-dir results
```

生成预测图：

```bash
python scripts/visualize_predictions.py \
  --checkpoint runs/03_unet_ce_dice/best.pt \
  --data-dir data/stanford_background \
  --split val \
  --out-dir results/predictions \
  --num-samples 8
```

## 7. 结果文件

每个实验目录：

```text
runs/<exp-name>/best.pt
runs/<exp-name>/last.pt
runs/<exp-name>/config.json
runs/<exp-name>/history.csv
runs/<exp-name>/metrics_best.json
```

汇总目录：

```text
results/summary.csv
results/summary.md
results/train_loss_curves.png
results/val_loss_curves.png
results/val_miou_curves.png
results/val_pixel_acc_curves.png
results/summary_bar.png
```

## 8. 注意

- 任务要求从零训练，所以不要加载 torchvision、timm 或 segmentation_models 的预训练权重。
- `scripts/model.py` 中 U-Net 是手写结构。
- `scripts/losses.py` 中 Dice Loss 是手动实现。
- 如果显存不够，把 batch size 从 8 改成 4 或 2，或者把 `base_channels` 从 32 改成 16。
