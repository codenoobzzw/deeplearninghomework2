# 深度学习期中作业 - 任务 1：Flowers102 图像分类微调实验

本项目用于完成任务 1：在 **102 Category Flower Dataset / Oxford Flowers102** 上微调 ImageNet 预训练 CNN，并完成 Baseline、超参数分析、预训练消融和注意力机制对比实验。

> 作业标题里写“宠物识别”，但正文指定的数据集是 `102 Category Flower Dataset`。本仓库按正文要求做花卉 102 类识别。

## 1. 本项目覆盖的作业要求

- **Baseline**：使用 ResNet-18 或 ResNet-34，替换最后输出层为 102 类分类器。
- **预训练微调**：使用 ImageNet 预训练权重初始化 backbone；新输出层随机初始化；输出层使用较大学习率，backbone 使用较小学习率。
- **超参数分析**：通过 `configs/experiment_plan.yaml` 跑不同学习率、训练轮数的组合。
- **预训练消融实验**：同样的 ResNet-18 从随机初始化训练，与 ImageNet 预训练模型对比。
- **注意力机制**：提供手动改造的 `se_resnet18` 和 `cbam_resnet18`，可与 Baseline 对比 Accuracy。
- **结果整理**：训练过程会保存 `history.csv/json`、`metrics_best.json`、`best.pt`；汇总脚本会生成 `results/summary.csv`、`summary.md` 和曲线图。
- **可视化日志**：支持 `wandb` 或 `swanlab`，也支持纯本地日志。

## 2. 目录结构

```text
.
├── configs/
│   └── experiment_plan.yaml       # 推荐实验计划：baseline、超参、随机初始化、SE、CBAM
├── scripts/
│   ├── check_env.py               # 检查 Python / PyTorch / CUDA / GPU
│   ├── prepare_data.py            # 下载并统计 Flowers102 数据集
│   ├── train.py                   # 单个实验训练入口
│   ├── evaluate.py                # 加载 checkpoint 做 val/test 评估
│   ├── run_experiments.py         # 顺序运行一组实验
│   ├── summarize_results.py       # 汇总所有实验结果并画图
│   └── plot_history.py            # 单次实验曲线图
├── src/flower_task1/
│   ├── data.py                    # 数据集与 transform
│   ├── models.py                  # ResNet / SE-ResNet / CBAM-ResNet
│   └── train_utils.py             # 训练辅助函数
├── PROMPT_FOR_CODEX.md            # 给服务器上 Codex 读取并执行的提示词
├── report_template.md             # 实验报告 Markdown 模板
├── requirements.txt
└── README.md
```

## 3. 环境配置

建议先在服务器上创建独立环境：

```bash
conda create -n dl_hw2_task1 python=3.10 -y
conda activate dl_hw2_task1
python -m pip install --upgrade pip
```

GPU 服务器最容易出错的是 PyTorch 与 CUDA 版本不匹配。推荐先到 PyTorch 官网根据服务器 CUDA 版本复制安装命令。如果不确定，先运行：

```bash
nvidia-smi
```

安装 PyTorch 后，再安装本项目依赖：

```bash
pip install -r requirements.txt
```

检查环境：

```bash
python scripts/check_env.py
```

如果输出中看到 `torch cuda available: True`，说明 PyTorch 可以使用 GPU。

## 4. 数据准备

直接使用 `torchvision.datasets.Flowers102` 自动下载官方 train / val / test split：

```bash
python scripts/prepare_data.py --data-dir data
```

成功后会生成：

```text
data/flowers102_info.json
```

里面包含 train、val、test 的样本数量和类别统计。

如果服务器不能联网，可以先在本地或其他机器下载 Flowers102，再把 `data/flowers-102` 相关目录拷贝到服务器的 `data/` 下，然后运行：

```bash
python scripts/prepare_data.py --data-dir data --no-download
```

## 5. 单次训练命令

### 5.1 Baseline：ImageNet 预训练 ResNet-18

```bash
python scripts/train.py \
  --data-dir data \
  --output-dir runs \
  --exp-name baseline_resnet18_pretrained \
  --model resnet18 \
  --pretrained \
  --epochs 30 \
  --batch-size 32 \
  --lr-head 1e-3 \
  --lr-backbone 1e-4 \
  --optimizer adamw \
  --scheduler cosine \
  --label-smoothing 0.1 \
  --tracker wandb \
  --wandb-mode offline \
  --amp
```

说明：`--lr-head` 对应新分类头学习率，`--lr-backbone` 对应预训练参数较小学习率，满足作业中“从零开始训练新的输出层，并对其余参数使用较小学习率微调”的要求。在 SE/CBAM 模型中，新增注意力模块也是随机初始化，因此代码会把注意力模块与分类头一起使用 `--lr-head`。

### 5.2 预训练消融：随机初始化 ResNet-18

```bash
python scripts/train.py \
  --data-dir data \
  --output-dir runs \
  --exp-name ablation_resnet18_random_init \
  --model resnet18 \
  --no-pretrained \
  --epochs 30 \
  --batch-size 32 \
  --lr-head 1e-3 \
  --lr-backbone 1e-3 \
  --optimizer adamw \
  --scheduler cosine \
  --label-smoothing 0.0 \
  --tracker wandb \
  --wandb-mode offline \
  --amp
```

### 5.3 注意力机制：SE-ResNet-18

```bash
python scripts/train.py \
  --data-dir data \
  --output-dir runs \
  --exp-name attention_se_resnet18_pretrained \
  --model se_resnet18 \
  --pretrained \
  --epochs 30 \
  --batch-size 32 \
  --lr-head 1e-3 \
  --lr-backbone 1e-4 \
  --tracker wandb \
  --wandb-mode offline \
  --amp
```

### 5.4 注意力机制：CBAM-ResNet-18

```bash
python scripts/train.py \
  --data-dir data \
  --output-dir runs \
  --exp-name attention_cbam_resnet18_pretrained \
  --model cbam_resnet18 \
  --pretrained \
  --epochs 30 \
  --batch-size 32 \
  --lr-head 1e-3 \
  --lr-backbone 1e-4 \
  --tracker wandb \
  --wandb-mode offline \
  --amp
```

## 6. 一键运行推荐实验计划

先做 1 个 epoch 的冒烟测试，确认数据、GPU、日志、checkpoint 都正常：

```bash
python scripts/run_experiments.py \
  --epochs-override 1 \
  --batch-size-override 8 \
  --tracker none \
  --amp
```

确认正常后，跑完整实验：

```bash
python scripts/run_experiments.py \
  --plan configs/experiment_plan.yaml \
  --data-dir data \
  --output-dir runs \
  --tracker wandb \
  --wandb-mode offline \
  --num-workers 4 \
  --amp
```

如果 GPU 显存不够，降低 batch size：

```bash
python scripts/run_experiments.py --batch-size-override 16 --tracker wandb --wandb-mode offline --amp
```

如果你只想用第 0 张 GPU：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_experiments.py --tracker wandb --wandb-mode offline --amp
```

## 7. wandb / swanlab 可视化

### wandb 离线模式

默认命令使用 `--wandb-mode offline`，不需要联网登录，也能在 `runs/<exp>/wandb/` 下保存日志。之后如果服务器能联网，可以同步：

```bash
wandb sync runs/<exp>/wandb/offline-run-*
```

### wandb 在线模式

```bash
wandb login
python scripts/run_experiments.py --tracker wandb --wandb-mode online --amp
```

### swanlab

```bash
python scripts/run_experiments.py --tracker swanlab --amp
```

报告里必须放训练集和验证集 loss 曲线，以及验证集 Accuracy 曲线。可以从 wandb / swanlab 截图，也可以把 `results/*.png` 放到报告中作为本地可视化补充。

## 8. 结果汇总

所有实验跑完后：

```bash
python scripts/summarize_results.py --runs-dir runs --out-dir results
```

生成：

```text
results/summary.csv
results/summary.md
results/summary_bar.png
results/val_accuracy_curves.png
results/train_loss_curves.png
results/val_loss_curves.png
```

报告中建议至少放一张表：

| 实验 | 模型 | 是否预训练 | lr_head | lr_backbone | epoch | Val Acc@1 | Test Acc@1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Baseline | ResNet-18 | 是 | 1e-3 | 1e-4 | 30 | 运行后填写 | 运行后填写 |
| 随机初始化 | ResNet-18 | 否 | 1e-3 | 1e-3 | 30 | 运行后填写 | 运行后填写 |
| 注意力 | SE-ResNet-18 | 是 | 1e-3 | 1e-4 | 30 | 运行后填写 | 运行后填写 |
| 注意力 | CBAM-ResNet-18 | 是 | 1e-3 | 1e-4 | 30 | 运行后填写 | 运行后填写 |

## 9. 模型权重与 GitHub 提交

每个实验目录都会保存：

```text
runs/<exp-name>/best.pt          # 验证集 Acc@1 最优 checkpoint
runs/<exp-name>/last.pt          # 最后一轮 checkpoint
runs/<exp-name>/config.json      # 实验设置
runs/<exp-name>/history.csv      # 每轮 train/val 指标
runs/<exp-name>/metrics_best.json
```

提交要求通常是：

1. 把代码上传到 public GitHub repo。
2. README 说明环境、训练、测试方法。
3. 把最好的 `best.pt` 上传到百度云 / Google Drive 等网盘。
4. 在 PDF 实验报告里写 GitHub repo 链接和模型权重网盘链接。

## 10. 常见问题

### 10.1 显存不够 / CUDA out of memory

把 batch size 从 32 改成 16 或 8：

```bash
python scripts/train.py ... --batch-size 16 --amp
```

### 10.2 数据集下载失败

先确认服务器能访问外网。如果不能，把本地下载好的 Flowers102 数据目录打包上传到服务器，然后使用 `--no-download`。

### 10.3 训练速度很慢

确认使用 GPU：

```bash
python scripts/check_env.py
```

如果 `torch cuda available: False`，需要重新安装匹配 CUDA 的 PyTorch。

### 10.4 wandb 不想联网

使用离线模式：

```bash
--tracker wandb --wandb-mode offline
```

### 10.5 需要更快得到初步结果

先把 `configs/experiment_plan.yaml` 里的 `epochs` 改成 10，确定趋势后再跑 30 或 50 epoch。
